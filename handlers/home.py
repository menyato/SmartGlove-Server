"""
handlers/home.py — HOME AUTOMATION feature: ESP32 relay control + WiFi-QR provisioning.

Architecture (per the chosen topology): the glove never talks to the ESP32
directly. The flow is:

    glove gesture -> hub.py -> ServerLink (existing TCP link, feature="home")
        -> HomeHandler.handle()
        -> plain HTTP request to the ESP32 relay board, over the home LAN

The ESP32 board itself is provisioned with WiFi credentials via a QR code:
this module generates a QR encoding a URL (http://192.168.4.1/provision?...,
see build_provision_url()) and serves it as a PNG on its own tiny HTTP
dashboard. Encoding a URL — not raw JSON — means any generic phone camera
app can scan it and offer "open in browser" directly: once the phone has
joined the ESP's RelayCtrl-Setup access point, tapping that link provisions
the board with no app and no manual typing. The OrangePi's own camera can
scan the exact same QR (features/home_automation.py, which accepts either
this URL form or legacy raw JSON) as part of its automated pairing flow.
This server never talks to the ESP until *after* it has joined the home
WiFi and registered itself here via POST /esp/register.

Actions received from OrangePi (feature="home"):
  toggle        — {"device_id"?, "relay": 1|2, "state": "on"|"off"|"toggle"}
  status        — {"device_id"?}  -> speaks current relay states
  list_devices  — {}              -> speaks how many relay boards are paired

HTTP dashboard on port 8090 (starts on first message, same pattern as
handlers/lidar.py's HTTP viewer):
  GET  /                    device list + links
  GET  /qr/<device_id>      setup QR PNG for that device id (generates on the fly)
  POST /esp/register        called by the ESP32 firmware after it joins WiFi
  GET  /metrics             relay toggle latency / error stats
"""

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from handlers.base import FeatureHandler

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEVICES_PATH  = os.path.join(SCRIPT_DIR, "home_devices.json")
METRICS_PATH  = os.path.join(SCRIPT_DIR, "home_metrics.jsonl")
HTTP_PORT     = 8090

PROVISION_TYPE     = "esp_relay_provision"
DEFAULT_DEVICE_ID  = "relay01"
DEFAULT_RELAY_COUNT = 2
ESP_REQUEST_TIMEOUT_S = 4.0
STALE_DEVICE_S = 120   # no heartbeat in this long -> treat IP as untrustworthy

# ── CONFIG (override via configure() from the CLI) ────────────────────────────
HOME_WIFI_SSID     = ""
HOME_WIFI_PASSWORD = ""
SERVER_HOST        = ""   # LAN IP the ESP should call back to; auto-detected if empty

_devices_lock  = threading.Lock()
_devices: dict = {}        # device_id -> {"ip","relay_count","fw","last_seen"}
_relay_cache: dict = {}    # device_id -> {1: bool, 2: bool}
_metrics_lock  = threading.Lock()

_http_started = False
_http_lock    = threading.Lock()
_actual_port  = HTTP_PORT


def configure(wifi_ssid: str | None = None, wifi_password: str | None = None,
              server_host: str | None = None) -> None:
    global HOME_WIFI_SSID, HOME_WIFI_PASSWORD, SERVER_HOST
    if wifi_ssid is not None:
        HOME_WIFI_SSID = wifi_ssid
    if wifi_password is not None:
        HOME_WIFI_PASSWORD = wifi_password
    if server_host is not None:
        SERVER_HOST = server_host


def _lan_ip() -> str:
    if SERVER_HOST:
        return SERVER_HOST
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "localhost"


# ── device registry persistence ────────────────────────────────────────────────

def _load_devices() -> None:
    global _devices
    try:
        with open(DEVICES_PATH, encoding="utf-8") as f:
            _devices = json.load(f)
    except (OSError, json.JSONDecodeError):
        _devices = {}


def _save_devices() -> None:
    try:
        with open(DEVICES_PATH, "w", encoding="utf-8") as f:
            json.dump(_devices, f, indent=2)
    except OSError:
        pass


def _register_device(device_id: str, ip: str, relay_count: int, fw: str) -> None:
    with _devices_lock:
        _devices[device_id] = {
            "ip": ip, "relay_count": relay_count, "fw": fw,
            "last_seen": time.time(),
        }
        _save_devices()
    print(f"[HOME] Device registered: {device_id} @ {ip} ({relay_count} relays, fw {fw})")


def _get_device(device_id: str) -> dict | None:
    with _devices_lock:
        d = _devices.get(device_id)
        return dict(d) if d else None


def _default_device_id() -> str | None:
    with _devices_lock:
        if not _devices:
            return None
        # most-recently-seen device wins if none was specified
        return max(_devices, key=lambda k: _devices[k].get("last_seen", 0))


# ── metrics ─────────────────────────────────────────────────────────────────────

def _log_metric(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _metrics_lock:
        try:
            with open(METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


# ── ESP HTTP calls ──────────────────────────────────────────────────────────────

def _esp_get(ip: str, path: str) -> tuple[dict | None, float]:
    """GET http://<ip><path>. Returns (json_or_None, elapsed_ms)."""
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"http://{ip}{path}", timeout=ESP_REQUEST_TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data, (time.time() - t0) * 1000
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None, (time.time() - t0) * 1000


# ── QR generation ────────────────────────────────────────────────────────────────

def build_provision_payload(device_id: str, relay_count: int) -> dict:
    return {
        "type":             PROVISION_TYPE,
        "ssid":             HOME_WIFI_SSID,
        "password":         HOME_WIFI_PASSWORD,
        "server_host":      _lan_ip(),
        "server_home_port": _actual_port,
        "device_id":        device_id,
        "relay_count":      relay_count,
    }


def build_provision_url(device_id: str, relay_count: int) -> str:
    """Encode the same fields as a URL pointing at the ESP's own setup access
    point (relay_controller.ino's GET /provision route). Scanning THIS with
    any generic phone camera/QR app offers "open in browser" directly — no
    app, no manual typing — and tapping it (once the phone has joined
    RelayCtrl-Setup) provisions the board immediately. This is what the QR
    endpoint below actually encodes; the OrangePi's automated pairing flow
    (features/home_automation.py) also accepts this format (it tries plain
    JSON first, then falls back to parsing a URL like this one)."""
    query = urlencode(build_provision_payload(device_id, relay_count))
    return f"http://192.168.4.1/provision?{query}"


def _generate_qr_png(content: str) -> "bytes | None":
    try:
        import io
        import qrcode
        img = qrcode.make(content)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        print("[HOME] qrcode package not installed — run: pip install qrcode[pil]")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP dashboard + ESP registration endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def start_http() -> None:
    """Public entry point — call this at server boot (from feature_server.py's
    main()) so the setup QR is servable immediately, before any pairing has
    happened. handle() also calls this on every message as a safety net
    (idempotent, guarded below), in case start_http() was never called."""
    _start_http()


def _start_http() -> None:
    global _http_started
    with _http_lock:
        if _http_started:
            return
        _http_started = True
    _load_devices()

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            p, qs = parsed.path.rstrip("/") or "/", parse_qs(parsed.query)
            if p == "/":
                self._index()
            elif p.startswith("/qr/"):
                self._qr(p[4:], qs.get("relays", [str(DEFAULT_RELAY_COUNT)])[0])
            elif p == "/metrics":
                self._metrics_page()
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path.rstrip("/") == "/esp/register":
                self._register()
            else:
                self.send_error(404)

        # ── pages ─────────────────────────────────────────────────────────────
        def _index(self):
            with _devices_lock:
                items = dict(_devices)
            now = time.time()
            rows = "".join(
                f"<tr><td>{did}</td><td>{d['ip']}</td><td>{d['relay_count']}</td>"
                f"<td>{d.get('fw','?')}</td>"
                f"<td>{'online' if now - d.get('last_seen',0) < STALE_DEVICE_S else 'stale'}</td>"
                f"<td><a href='/qr/{did}'>Re-provision QR</a></td></tr>"
                for did, d in items.items()
            ) or "<tr><td colspan='6'>No relay boards paired yet.</td></tr>"
            self._html(f"""<!DOCTYPE html><html>
<head><title>Home Automation</title>{_CSS}</head>
<body><h2>Paired Relay Boards</h2>
<table><tr><th>Device</th><th>IP</th><th>Relays</th><th>FW</th><th>Status</th><th>QR</th></tr>{rows}</table>
<p>New device? Generate its setup QR before pairing:
<a href="/qr/{DEFAULT_DEVICE_ID}">/qr/{DEFAULT_DEVICE_ID}</a></p>
<p><a href="/metrics">Relay latency metrics →</a></p>
</body></html>""")

        def _qr(self, device_id: str, relays: str):
            device_id = device_id or DEFAULT_DEVICE_ID
            try:
                relay_count = int(relays)
            except ValueError:
                relay_count = DEFAULT_RELAY_COUNT
            url = build_provision_url(device_id, relay_count)
            png = _generate_qr_png(url)
            if png is None:
                payload = build_provision_payload(device_id, relay_count)
                self._html("<h2>qrcode package not installed on the server.</h2>"
                            "<pre>pip install qrcode[pil]</pre>"
                            f"<pre>{json.dumps(payload, indent=2)}</pre>")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def _metrics_page(self):
            lines = []
            try:
                with open(METRICS_PATH, encoding="utf-8") as f:
                    lines = [json.loads(l) for l in f if l.strip()]
            except (OSError, json.JSONDecodeError):
                pass
            if not lines:
                body = "<p>No relay activity logged yet.</p>"
            else:
                keys = list(lines[-1].keys())
                hdrs = "".join(f"<th>{k}</th>" for k in keys)
                rows = "".join(
                    "<tr>" + "".join(f"<td>{r.get(k,'')}</td>" for k in keys) + "</tr>"
                    for r in lines[-50:]
                )
                body = f"<table><tr>{hdrs}</tr>{rows}</table>"
            self._html(f"""<!DOCTYPE html><html>
<head><title>Home Automation Metrics</title>{_CSS}</head>
<body><h2>Relay Command Latency (last {min(50,len(lines))} of {len(lines)})</h2>
{body}<p><a href="/">← Devices</a></p></body></html>""")

        def _register(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                device_id   = str(body["device_id"])
                ip          = str(body["ip"])
                relay_count = int(body.get("relay_count", DEFAULT_RELAY_COUNT))
                fw          = str(body.get("fw", "?"))
            except (KeyError, ValueError, json.JSONDecodeError):
                self.send_response(400)
                self.end_headers()
                return
            _register_device(device_id, ip, relay_count, fw)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def _html(self, body: str):
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *_): pass

    def _run():
        global _actual_port
        for port in range(HTTP_PORT, HTTP_PORT + 10):
            try:
                srv = HTTPServer(("0.0.0.0", port), _H)
                _actual_port = port
                print(f"\n[HOME] === HTTP dashboard started ===")
                print(f"[HOME]  Devices: http://{_lan_ip()}:{port}/")
                print(f"[HOME]  Setup QR: http://{_lan_ip()}:{port}/qr/{DEFAULT_DEVICE_ID}")
                print(f"[HOME] ===============================\n")
                srv.serve_forever()
                return
            except OSError:
                continue
        print(f"[HOME] HTTP dashboard: no free port in {HTTP_PORT}-{HTTP_PORT+9}.")

    threading.Thread(target=_run, daemon=True, name="home-http").start()


_CSS = """<style>
body{font-family:sans-serif;margin:2em;color:#222}
h2{color:#333}
table{border-collapse:collapse;width:100%;max-width:900px}
th,td{padding:6px 14px;border:1px solid #ccc;text-align:left}
th{background:#f4f4f4}
a{color:#07c}
</style>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Feature handler
# ═══════════════════════════════════════════════════════════════════════════════

class HomeHandler(FeatureHandler):
    name = "home"

    def handle(self, msg: dict) -> dict:
        _start_http()
        action = msg.get("action", "status")
        if action == "toggle":
            return self._toggle(msg)
        if action == "status":
            return self._status(msg)
        if action == "list_devices":
            return self._list_devices(msg)
        if action == "get_relay_count":
            return self._get_relay_count(msg)
        return {"tts": "", "quit": False}

    def _resolve_device(self, msg: dict) -> tuple[str | None, dict | None]:
        device_id = msg.get("device_id") or _default_device_id()
        if device_id is None:
            return None, None
        return device_id, _get_device(device_id)

    def _get_relay_count(self, msg: dict) -> dict:
        """Used by hub.py at boot to build exactly as many gesture-bindable
        RelaySwitch features as the paired board actually has, instead of a
        hardcoded number. Silent — never spoken."""
        _device_id, device = self._resolve_device(msg)
        count = device["relay_count"] if device else DEFAULT_RELAY_COUNT
        return {"tts": "", "quit": False, "relay_count": count}

    def _toggle(self, msg: dict) -> dict:
        device_id, device = self._resolve_device(msg)
        relay = int(msg.get("relay", 1))
        state = msg.get("state", "toggle")

        if device is None:
            _log_metric({"action": "toggle", "device_id": device_id, "relay": relay,
                         "ok": False, "error": "no_device"})
            return {"tts": "No relay board paired yet. Say pair to set one up.", "quit": False}

        stale = time.time() - device.get("last_seen", 0) > STALE_DEVICE_S
        resp, elapsed_ms = _esp_get(device["ip"], f"/relay?ch={relay}&state={state}")
        ok = resp is not None and resp.get("ok")
        _log_metric({
            "action": "toggle", "device_id": device_id, "relay": relay,
            "requested_state": state, "esp_latency_ms": round(elapsed_ms, 1),
            "ok": ok, "stale_registration": stale,
        })

        if not ok:
            hint = " The board's last known address may be out of date." if stale else ""
            return {"tts": f"Could not reach relay {relay}.{hint}", "quit": False}

        new_state = resp.get("state", "?")
        with _devices_lock:
            _relay_cache.setdefault(device_id, {})[relay] = (new_state == "on")
        return {"tts": f"Relay {relay} turned {new_state}.", "quit": False}

    def _status(self, msg: dict) -> dict:
        device_id, device = self._resolve_device(msg)
        if device is None:
            return {"tts": "No relay board paired yet.", "quit": False}

        resp, elapsed_ms = _esp_get(device["ip"], "/status")
        _log_metric({"action": "status", "device_id": device_id,
                     "esp_latency_ms": round(elapsed_ms, 1), "ok": resp is not None})
        if resp is None:
            return {"tts": "Could not reach the relay board.", "quit": False}

        parts = [f"relay {k[5:]} is {v}" for k, v in sorted(resp.items())]
        return {"tts": ", ".join(parts).capitalize() + ".", "quit": False}

    def _list_devices(self, msg: dict) -> dict:
        with _devices_lock:
            n = len(_devices)
        if n == 0:
            return {"tts": "No relay boards paired yet.", "quit": False}
        return {"tts": f"{n} relay board{'s' if n != 1 else ''} paired.", "quit": False}
