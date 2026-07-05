"""
handlers/lidar.py — Server handler for all LiDAR feature messages.

Actions received from OrangePi:
  map_save    — permanent room PNG, saved to lidar_maps/<room>.png
  map_update  — live/in-progress map PNG for 'live_map' view (silent, no TTS)
  pose_update — current SLAM pose (x_m, y_m); overlaid as a red dot
  report      — session results JSON, saved to lidar_reports/<session>_<mode>.json

HTTP viewer on port 8080 (starts on first message):
  /maps                   list saved rooms
  /live?room=X            auto-refreshing live view with position dot
  /map/<room>.png         latest PNG with pose overlay
  /reports                list all saved reports
  /report/<filename>      formatted HTML view of one report
"""

import base64
import json
import math
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from handlers.base import FeatureHandler

MAPS_DIR    = Path(__file__).parent / "lidar_maps"
REPORTS_DIR = Path(__file__).parent / "lidar_reports"
HTTP_PORT   = 8080

# Must match lidar_nav.py / slam_engine.py
_MAP_RES  = 0.05
_MAP_ORIG = 15.0   # = SLAM_SIZE_M / 2

_live: dict  = {}   # room → (col, row)
_lock = threading.Lock()
_feature_state: dict = {}   # tracks env/ocr/money state pushed by features

_http_started = False
_http_lock    = threading.Lock()
_actual_port  = HTTP_PORT   # updated to real port when server starts


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP viewer
# ═══════════════════════════════════════════════════════════════════════════════

def _start_http() -> None:
    global _http_started
    with _http_lock:
        if _http_started:
            return
        _http_started = True
    MAPS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            p, qs  = parsed.path.rstrip("/") or "/", parse_qs(parsed.query)
            if   p in ("/", "/maps"):                      self._index()
            elif p.startswith("/map/") and p.endswith(".png"):
                self._serve_png(p[5:-4])
            elif p == "/live":                             self._live_page(qs.get("room",[None])[0])
            elif p == "/reports":                          self._reports()
            elif p.startswith("/report/"):                 self._report_page(p[8:])
            elif p == "/metrics":                          self._metrics_page()
            elif p == "/state":                            self._state_page()
            else:                                          self.send_error(404)

        # ── pages ─────────────────────────────────────────────────────────────
        def _index(self):
            rooms = sorted({p.stem for p in MAPS_DIR.glob("*.png")
                            if not p.stem.endswith("_live")})
            rows = "".join(
                f'<tr><td>{r}</td>'
                f'<td><a href="/live?room={r}">Live view</a></td>'
                f'<td><a href="/map/{r}.png">PNG</a></td></tr>'
                for r in rooms
            ) or '<tr><td colspan="3">No rooms saved yet.</td></tr>'
            self._html(f"""<!DOCTYPE html><html>
<head><title>Lidar Maps</title>{_CSS}</head>
<body><h2>Saved Rooms</h2>
<table><tr><th>Room</th><th>View</th><th>PNG</th></tr>{rows}</table>
<p><a href="/reports">Session reports →</a> &nbsp;|&nbsp;
   <a href="/metrics">Feature metrics →</a> &nbsp;|&nbsp;
   <a href="/state">Glove state →</a></p>
</body></html>""")

        def _live_page(self, room):
            if not room:
                rooms = sorted({p.stem for p in MAPS_DIR.glob("*.png")
                                if not p.stem.endswith("_live")})
                room = rooms[-1] if rooms else None
            if not room:
                self._html("<h2>No rooms saved yet.</h2>"); return
            self._html(f"""<!DOCTYPE html><html>
<head><title>Live: {room}</title></head>
<body style="margin:0;background:#111;color:#eee;text-align:center">
<h2 style="margin:.5em 0">Live — {room.replace('_',' ')}</h2>
<img id="m" src="/map/{room}.png"
     style="max-width:95vw;max-height:85vh;image-rendering:pixelated;border:2px solid #444">
<p><a href="/maps" style="color:#aaf">All rooms</a>
 &nbsp;|&nbsp; <a href="/reports" style="color:#aaf">Reports</a>
 &nbsp;|&nbsp; <span style="color:#888">● = current position</span></p>
<script>
setInterval(function(){{
  document.getElementById('m').src='/map/{room}.png?_='+Date.now();
}},1500);
</script></body></html>""")

        def _serve_png(self, room):
            if not re.fullmatch(r'[\w\-]+', room or ''):
                self.send_error(400); return
            base = MAPS_DIR / f"{room}.png"
            if not base.exists():
                # return a small grey "waiting" placeholder so the live page
                # doesn't show a broken-image icon before data arrives
                png = _waiting_png()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(png)
                return
            with _lock:
                pos = _live.get(room)
            png = _overlay(base.read_bytes(), pos)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(png)

        # ── report pages ──────────────────────────────────────────────────────
        def _reports(self):
            files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
            rows = "".join(
                f'<tr><td><a href="/report/{f.name}">{f.stem}</a></td>'
                f'<td>{f.stat().st_size//1024} KB</td></tr>'
                for f in files
            ) or '<tr><td colspan="2">No reports yet.</td></tr>'
            self._html(f"""<!DOCTYPE html><html>
<head><title>Lidar Reports</title>{_CSS}</head>
<body><h2>Session Reports</h2>
<table><tr><th>Report</th><th>Size</th></tr>{rows}</table>
<p><a href="/maps">← Maps</a></p>
</body></html>""")

        def _report_page(self, filename):
            path = REPORTS_DIR / filename
            if not path.exists():
                self.send_error(404); return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                self._html(f"<pre>Error reading report: {e}</pre>"); return
            mode = data.get("mode", "?")
            html = _render_report(mode, data)
            self._html(html)

        def _metrics_page(self):
            import os, json as _json
            handlers_dir = Path(__file__).parent
            sections = []
            for name, label in [
                ("feature_metrics.jsonl", "All Features — Latency &amp; Response Time"),
                ("client_metrics.jsonl",  "Client-Side (OrangePi) Load Times"),
                ("ocr_metrics.jsonl",     "OCR"),
                ("money_metrics.jsonl",   "Money"),
                ("home_metrics.jsonl",    "Home Automation"),
                ("tts_metrics.jsonl",     "TTS"),
            ]:
                path = handlers_dir / name
                if not path.exists():
                    sections.append(f"<h3>{label}</h3><p>No data yet.</p>")
                    continue
                lines = []
                try:
                    with open(path, encoding="utf-8") as f:
                        lines = [_json.loads(l) for l in f if l.strip()]
                except Exception:
                    pass
                if not lines:
                    sections.append(f"<h3>{label}</h3><p>No entries yet.</p>")
                    continue
                keys = list(lines[-1].keys())
                hdrs = "".join(f"<th>{k}</th>" for k in keys)
                rows = "".join(
                    "<tr>" + "".join(f"<td>{r.get(k,'')}</td>" for k in keys) + "</tr>"
                    for r in lines[-50:]
                )
                sections.append(
                    f"<h3>{label} — last {min(50,len(lines))} of {len(lines)} entries</h3>"
                    f"<div style='overflow-x:auto'><table><tr>{hdrs}</tr>{rows}</table></div>"
                )
            body = f"""<!DOCTYPE html><html>
<head><title>Feature Metrics</title>{_CSS}</head>
<body><h2>Feature Performance Metrics</h2>
{"".join(sections)}
<p><a href="/maps">← Maps</a> | <a href="/reports">Reports</a> | <a href="/state">Glove State</a></p>
</body></html>"""
            self._html(body)

        def _state_page(self):
            import time as _time
            with _lock:
                fs = dict(_feature_state)
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for item in fs.values()
                for k, v in item.items()
            ) or "<tr><td colspan='2'>No active feature state.</td></tr>"
            body = f"""<!DOCTYPE html><html>
<head><title>Glove State</title>{_CSS}</head>
<body><h2>Live Glove Feature State</h2>
<p style='color:#888'>Updates when features send state changes. Refresh to see latest.</p>
<table><tr><th>Key</th><th>Value</th></tr>{rows}</table>
<p><a href="/maps">← Maps</a> | <a href="/metrics">Metrics</a></p>
</body></html>"""
            self._html(body)

        def _html(self, body: str):
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *_): pass

    def _run():
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            lan_ip = "localhost"
        for port in range(HTTP_PORT, HTTP_PORT + 10):
            try:
                srv = HTTPServer(("0.0.0.0", port), _H)
                print(f"\n[LIDAR] === HTTP viewer started ===")
                print(f"[LIDAR]  Maps:    http://{lan_ip}:{port}/maps")
                print(f"[LIDAR]  Live:    http://{lan_ip}:{port}/live")
                print(f"[LIDAR]  Reports: http://{lan_ip}:{port}/reports")
                print(f"[LIDAR] ============================\n")
                srv.serve_forever()
                return
            except OSError:
                continue
        print(f"[LIDAR] HTTP viewer: no free port in {HTTP_PORT}-{HTTP_PORT+9}. "
              "Run as Administrator or free a port.")

    threading.Thread(target=_run, daemon=True, name="lidar-http").start()


# ─── shared CSS ───────────────────────────────────────────────────────────────
_CSS = """<style>
body{font-family:sans-serif;margin:2em;color:#222}
h2{color:#333}
table{border-collapse:collapse;width:100%;max-width:900px}
th,td{padding:6px 14px;border:1px solid #ccc;text-align:left}
th{background:#f4f4f4}
a{color:#07c}
</style>"""


# ─── helpers ──────────────────────────────────────────────────────────────────

# Minimal 1×1 grey PNG — returned while waiting for first data from OrangePi.
# Avoids broken-image icons in the browser live view.
_WAITING_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAABmJLR0QA/wD/AP+gvaeTAAAAB3RJ"
    "TUUH6AQKDxkPmNe9BAAAACpJREFUeNrtwTEBAAAAwqD1T20Hb6AAAAAAAAAAAAAAAAAAAAAAwA8m"
    "AAABqLgHQAAAABJRU5ErkJggg=="
)

def _waiting_png() -> bytes:
    """300×300 dark-grey PNG with 'Waiting for data…' text if cv2 available."""
    try:
        import cv2, numpy as np
        img = np.full((300, 300, 3), 30, dtype=np.uint8)
        cv2.putText(img, "Waiting for data...", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
        ok, buf = cv2.imencode(".png", img)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return base64.b64decode(_WAITING_PNG_B64)


def _world_to_px(x_m: float, y_m: float):
    return int((x_m + _MAP_ORIG) / _MAP_RES), int((y_m + _MAP_ORIG) / _MAP_RES)


def _overlay(base_png: bytes, pos) -> bytes:
    if pos is None:
        return base_png
    col, row = pos
    try:
        import cv2, numpy as np
        arr = np.frombuffer(base_png, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return base_png
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = rgb.shape[:2]
        if 0 <= col < w and 0 <= row < h:
            cv2.circle(rgb, (col, row), 7,  (0, 0, 255), -1)
            cv2.circle(rgb, (col, row), 10, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".png", rgb)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return base_png


# ─── report HTML renderers ────────────────────────────────────────────────────

def _render_report(mode: str, data: dict) -> str:
    sid  = data.get("session_id", "?")
    d    = data.get("data", data)      # unwrap if nested
    if mode == "obstacles":
        return _render_obstacles(sid, d)
    if mode == "mapping":
        return _render_mapping(sid, d)
    if mode == "navigate":
        return _render_navigate(sid, d)
    return f"<pre>{json.dumps(data, indent=2)}</pre>"


def _render_obstacles(sid: str, d: dict) -> str:
    def stat_row(label, s):
        if not s or not s.get("samples"):
            return f"<tr><td>{label}</td><td colspan='3'>no readings</td></tr>"
        return (f"<tr><td>{label}</td>"
                f"<td>{s['min_m']} m</td><td>{s['avg_m']} m</td><td>{s['max_m']} m</td></tr>")

    mf = d.get("motor_fires", {})
    events = d.get("events", [])
    event_rows = "".join(
        f"<tr><td>{e['t']}</td><td>{e['front_m'] or '—'}</td>"
        f"<td>{e['left_m'] or '—'}</td><td>{e['right_m'] or '—'}</td>"
        f"<td>{e['motor'] or ''}</td></tr>"
        for e in events[:200]   # cap to 200 rows in HTML
    )
    return f"""<!DOCTYPE html><html>
<head><title>Obstacle Report — {sid}</title>{_CSS}</head>
<body>
<h2>Obstacle Detection Report</h2>
<p><b>Session:</b> {sid} &nbsp;|&nbsp;
   <b>Duration:</b> {d.get('duration_s','?')} s &nbsp;|&nbsp;
   <b>Samples:</b> {d.get('total_samples','?')}</p>

<h3>Distance Statistics</h3>
<table>
  <tr><th>Sector</th><th>Min</th><th>Avg</th><th>Max</th></tr>
  {stat_row('Front (MT3)', d.get('front',{}))}
  {stat_row('Left  (MT2)', d.get('left',{}))}
  {stat_row('Right (MT1)', d.get('right',{}))}
</table>

<h3>Motor Activations</h3>
<table>
  <tr><th>Motor</th><th>Fires</th></tr>
  <tr><td>MT1 (right)</td><td>{mf.get('MT1',0)}</td></tr>
  <tr><td>MT2 (left)</td><td>{mf.get('MT2',0)}</td></tr>
  <tr><td>MT3 (front)</td><td>{mf.get('MT3',0)}</td></tr>
</table>

<h3>Event Log (first 200)</h3>
<table>
  <tr><th>Time (s)</th><th>Front (m)</th><th>Left (m)</th><th>Right (m)</th><th>Motor</th></tr>
  {event_rows}
</table>
<p><a href="/reports">← All reports</a></p>
</body></html>"""


def _render_mapping(sid: str, d: dict) -> str:
    kf_rows = "".join(
        f"<tr><td>{k['kf_id']}</td><td>{k['t']}</td>"
        f"<td>{k['x']}</td><td>{k['y']}</td><td>{k.get('yaw_deg','?')}°</td></tr>"
        for k in d.get("keyframe_log", [])
    )
    room = d.get('room_name', '?')
    map_link = f'<p><a href="/live?room={room}">View live map →</a></p>' if room != '?' else ''
    return f"""<!DOCTYPE html><html>
<head><title>Mapping Report — {sid}</title>{_CSS}</head>
<body>
<h2>Mapping Report</h2>
<p><b>Session:</b> {sid} &nbsp;|&nbsp;
   <b>Room:</b> {room} &nbsp;|&nbsp;
   <b>Duration:</b> {d.get('duration_s','?')} s</p>

<h3>Summary</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Keyframes</td><td>{d.get('keyframes','?')}</td></tr>
  <tr><td>Distance walked</td><td>{d.get('distance_m','?')} m</td></tr>
  <tr><td>Occupied cells</td><td>{d.get('occupied_cells','?')}</td></tr>
  <tr><td>Loop closures</td><td>{d.get('loop_closures','?')}</td></tr>
</table>
{map_link}

<h3>Keyframe Log</h3>
<table>
  <tr><th>ID</th><th>Time (s)</th><th>X (m)</th><th>Y (m)</th><th>Yaw</th></tr>
  {kf_rows or '<tr><td colspan="5">No keyframes</td></tr>'}
</table>
<p><a href="/reports">← All reports</a></p>
</body></html>"""


def _render_navigate(sid: str, d: dict) -> str:
    arrived = "✓ Yes" if d.get("arrived") else "✗ No"
    pose_rows = "".join(
        f"<tr><td>{p['t']}</td><td>{p['x']}</td><td>{p['y']}</td>"
        f"<td>{p['dist_m']}</td><td>{p.get('bearing_deg','?')}°</td></tr>"
        for p in d.get("pose_log", [])
    )
    room = d.get("room", "?")
    map_link = f'<p><a href="/live?room={room}">View live map →</a></p>' if room != '?' else ''
    return f"""<!DOCTYPE html><html>
<head><title>Navigate Report — {sid}</title>{_CSS}</head>
<body>
<h2>Navigation Report</h2>
<p><b>Session:</b> {sid} &nbsp;|&nbsp;
   <b>Target room:</b> {room} &nbsp;|&nbsp;
   <b>Duration:</b> {d.get('duration_s','?')} s</p>

<h3>Summary</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Arrived</td><td>{arrived}</td></tr>
  <tr><td>Start distance</td><td>{d.get('start_dist_m','?')} m</td></tr>
  <tr><td>End distance</td><td>{d.get('end_dist_m','?')} m</td></tr>
  <tr><td>Path length</td><td>{d.get('path_length_m','?')} m</td></tr>
</table>
{map_link}

<h3>Pose Log</h3>
<table>
  <tr><th>Time (s)</th><th>X (m)</th><th>Y (m)</th>
      <th>Distance to target (m)</th><th>Bearing (°)</th></tr>
  {pose_rows or '<tr><td colspan="5">No poses recorded</td></tr>'}
</table>
<p><a href="/reports">← All reports</a></p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Feature handler
# ═══════════════════════════════════════════════════════════════════════════════

class LidarHandler(FeatureHandler):
    name = "lidar"

    def handle(self, msg: dict) -> dict:
        _start_http()
        action = msg.get("action", "map_save")

        if action == "map_save":
            return self._map_save(msg, announce=True)
        if action == "map_update":
            return self._map_save(msg, announce=False)
        if action == "pose_update":
            return self._pose_update(msg)
        if action == "report":
            return self._report(msg)
        if action == "feature_state":
            return self._feature_state(msg)
        return {"tts": "", "quit": False}

    # ── map save / update ──────────────────────────────────────────────────────
    def _map_save(self, msg: dict, announce: bool) -> dict:
        room  = msg.get("room_name", "room")
        b64   = msg.get("frame", "")
        if not b64:
            return {"tts": "", "quit": False}
        try:
            png = base64.b64decode(b64)
        except Exception as e:
            return {"tts": f"Bad map data: {e}", "quit": False}

        MAPS_DIR.mkdir(exist_ok=True)
        out = MAPS_DIR / f"{room}.png"
        out.write_bytes(png)
        if announce:
            print(f"[LIDAR] Saved {out.name} ({len(png)//1024} KB)"
                  f"  →  http://localhost:{HTTP_PORT}/live?room={room}")
            return {"tts": f"Map saved for {room.replace('_',' ')}.", "quit": False}
        else:
            print(f"[LIDAR] Live map updated: {out.name} ({len(png)//1024} KB)")
            return {"tts": "", "quit": False}

    # ── pose update ───────────────────────────────────────────────────────────
    def _pose_update(self, msg: dict) -> dict:
        room = msg.get("room_name", "")
        x_m  = float(msg.get("x", 0.0))
        y_m  = float(msg.get("y", 0.0))
        if room and (MAPS_DIR / f"{room}.png").exists():
            col, row = _world_to_px(x_m, y_m)
            with _lock:
                _live[room] = (col, row)
        return {"tts": "", "quit": False}

    # ── report ─────────────────────────────────────────────────────────────────
    def _report(self, msg: dict) -> dict:
        mode       = msg.get("mode", "unknown")
        session_id = msg.get("session_id", "session")
        data       = msg.get("data", {})

        REPORTS_DIR.mkdir(exist_ok=True)
        filename = f"{session_id}_{mode}.json"
        path     = REPORTS_DIR / filename
        payload  = {"mode": mode, "session_id": session_id, "data": data}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[LIDAR] Report saved: {filename}"
              f"  →  http://localhost:{HTTP_PORT}/report/{filename}")
        return {"tts": f"{mode} report saved.", "quit": False}

    def _feature_state(self, msg: dict) -> dict:
        key = msg.get("feature_name", "unknown")
        with _lock:
            _feature_state[key] = {k: v for k, v in msg.items()
                                   if k not in ("feature", "action")}
        print(f"[LIDAR] Feature state: {key} → {msg.get('state','?')}")
        return {"tts": "", "quit": False}
