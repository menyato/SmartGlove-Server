#!/usr/bin/env python3
"""
feature_server.py — THE GENERAL SERVER on the laptop.

It connects to the OrangePi glove hub (one socket), reads the `feature` field
on every message, and routes the message to that feature's helper. Each helper
is a self-contained file under handlers/ that runs its own logic (money ->
handlers/money.py with YOLO currency detection). The reply text is voiced to a
natural WAV via the shared tts module and sent back.

Add a feature:
  1. write handlers/<name>.py with a FeatureHandler subclass,
  2. register its class in FEATURE_HANDLERS below.
No change to this loop is needed.

Run from server_app/:

    python feature_server.py --host 0.0.0.0 --port 9000 ^
        --tts auto --piper-model C:\\path\\voice.onnx --sapi-voice Zira ^
        --model C:\\path\\best.pt
"""

import argparse
import base64
import signal
import socket
import sys
import threading

import time

# Force UTF-8 stdout/stderr regardless of the Windows console's default
# codepage (often cp1252, which can't encode arrows/box-drawing characters
# used in log messages throughout this codebase — a bare print() with one of
# those would otherwise raise UnicodeEncodeError and silently kill whichever
# thread hit it, e.g. a handler's lazily-started HTTP dashboard thread).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import tts
from protocol import recv_msg, send_msg
from handlers import money, ocr, home, env
from handlers.tts_handler import TTSHandler
from handlers.lidar import LidarHandler
from handlers.metrics_handler import MetricsHandler

# ── Glove state tracker ───────────────────────────────────────────────────────

class GloveState:
    """Server-side tracking of every connected hub and its active feature."""
    _lock   = threading.Lock()
    _hubs: dict  = {}   # addr_str -> hub_info dict
    _counts: dict = {}  # feature -> {"scans": int, "errors": int}

    @classmethod
    def connect(cls, addr) -> None:
        key = f"{addr[0]}:{addr[1]}"
        with cls._lock:
            cls._hubs[key] = {
                "addr":         key,
                "connected_at": time.time(),
                "feature":      None,
                "feature_at":   None,
                "msg_count":    0,
            }

    @classmethod
    def disconnect(cls, addr) -> None:
        key = f"{addr[0]}:{addr[1]}"
        with cls._lock:
            cls._hubs.pop(key, None)

    @classmethod
    def touch(cls, addr, feature: str) -> None:
        key = f"{addr[0]}:{addr[1]}"
        with cls._lock:
            hub = cls._hubs.get(key)
            if hub is None:
                return
            if hub.get("feature") != feature:
                hub["feature"]    = feature
                hub["feature_at"] = time.time()
            hub["msg_count"] += 1
            cnt = cls._counts.setdefault(feature, {"scans": 0, "errors": 0})
            cnt["scans"] += 1

    @classmethod
    def error(cls, addr, feature: str) -> None:
        with cls._lock:
            cnt = cls._counts.setdefault(feature, {"scans": 0, "errors": 0})
            cnt["errors"] += 1

    @classmethod
    def snapshot(cls) -> dict:
        with cls._lock:
            now = time.time()
            hubs = {}
            for key, hub in cls._hubs.items():
                s = dict(hub)
                s["uptime_s"] = round(now - hub["connected_at"])
                if hub["feature_at"]:
                    s["feature_age_s"] = round(now - hub["feature_at"])
                hubs[key] = s
            return {"hubs": hubs, "feature_counts": dict(cls._counts)}


# ── TTS metrics ───────────────────────────────────────────────────────────────

import json as _json
import os as _os

_TTS_METRICS_PATH = _os.path.join(_os.path.dirname(__file__), "handlers", "tts_metrics.jsonl")
_tts_metrics_lock = threading.Lock()


def _log_tts_metric(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _tts_metrics_lock:
        try:
            with open(_TTS_METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
        except OSError:
            pass


# ── Consolidated per-feature request metrics ──────────────────────────────────
# One entry per dispatched message, for EVERY feature, with no per-handler code
# changes required. `latency_ms` = wall-clock time between the client stamping
# the message (ServerLink.send injects "_t_sent") and this server receiving it
# — i.e. network + queueing time. `processing_ms` = time spent inside the
# handler's own handle() call — i.e. the feature's actual response time.
_FEATURE_METRICS_PATH = _os.path.join(_os.path.dirname(__file__), "handlers", "feature_metrics.jsonl")
_feature_metrics_lock = threading.Lock()


def _log_feature_metric(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _feature_metrics_lock:
        try:
            with open(_FEATURE_METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
        except OSError:
            pass


# feature name -> handler class. One instance per connection per feature.
FEATURE_HANDLERS = {
    "money":   money.MoneyHandler,
    "ocr":     ocr.OCRHandler,
    "tts":     TTSHandler,
    "lidar":   LidarHandler,
    "home":    home.HomeHandler,
    "env":     env.EnvHandler,
    "metrics": MetricsHandler,
}


def _send_with_audio(conn: socket.socket, resp: dict) -> None:
    """Voice the reply text to a WAV (base64) and send the response."""
    text = resp.get("tts", "")
    if text:
        tts.local_speak(text)
        t0 = time.time()
        wav, engine = tts.synth_wav(text)
        latency_ms = int((time.time() - t0) * 1000)
        if wav:
            resp["audio"] = base64.b64encode(wav).decode("ascii")
            print(f"[TTS] rendered via {engine} ({len(wav)/1024:.1f} KB)")
            _log_tts_metric({
                "chars":      len(text),
                "engine":     engine,
                "wav_kb":     round(len(wav) / 1024, 1),
                "latency_ms": latency_ms,
            })
        else:
            print(f"[TTS] no PC audio ({engine}); client will use espeak.")
    send_msg(conn, resp)


def handle_client(conn: socket.socket, addr) -> None:
    print(f"[+] Hub connected: {addr}")
    GloveState.connect(addr)
    handlers: dict[str, object] = {}
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break

            feature = msg.get("feature", "money")
            GloveState.touch(addr, feature)
            cls = FEATURE_HANDLERS.get(feature)
            if cls is None:
                _send_with_audio(conn, {"tts": f"Unknown feature {feature}.", "quit": False})
                continue

            handler = handlers.get(feature)
            if handler is None:
                handler = cls()
                handlers[feature] = handler
                print(f"[ROUTE] new {feature} session for {addr}")

            t_recv = time.time()
            t_sent = msg.get("_t_sent")
            latency_ms = round((t_recv - t_sent) * 1000, 1) if t_sent else None

            try:
                resp = handler.handle(msg)
                ok = True
            except Exception as he:
                import traceback
                traceback.print_exc()
                GloveState.error(addr, feature)
                resp = {"tts": f"{feature} error: {he}", "quit": False}
                ok = False
            processing_ms = round((time.time() - t_recv) * 1000, 1)

            _log_feature_metric({
                "feature": feature, "type": msg.get("type") or msg.get("action"),
                "latency_ms": latency_ms, "processing_ms": processing_ms,
                "total_ms": (latency_ms + processing_ms) if latency_ms is not None else None,
                "ok": ok,
            })

            _send_with_audio(conn, resp)

            if resp.get("quit"):
                handlers.pop(feature, None)
                print(f"[ROUTE] {feature} session ended for {addr}")
    except Exception as e:
        print(f"[!] Client error: {e}")
    finally:
        GloveState.disconnect(addr)
        conn.close()
        print(f"[-] Hub disconnected: {addr}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smart Glove general feature server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=9000, type=int)
    # shared TTS
    ap.add_argument("--tts", default="auto", choices=["auto", "piper", "sapi"])
    ap.add_argument("--piper-model", default="")
    ap.add_argument("--sapi-voice", default="Zira")
    ap.add_argument("--sapi-rate", default=0, type=int)
    ap.add_argument("--ocr-engine", default="easyocr", choices=["easyocr", "paddleocr"],
                    help="OCR engine for the Book Reader feature. 'paddleocr' runs in an "
                         "isolated subprocess and falls back to easyocr automatically if "
                         "it can't start on this machine.")
    # money feature tuning
    ap.add_argument("--model", default=None, help="YOLO currency model path (.pt)")
    ap.add_argument("--detect-conf", default=None, type=float)
    ap.add_argument("--box-conf", default=None, type=float)
    ap.add_argument("--reliable-conf", default=None, type=float,
                    help="0 = off (original). >0 (e.g. 0.55) flags weak reads for rescan.")
    ap.add_argument("--imgsz", default=None, type=int)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--blur-radius", default=None, type=int)
    # home automation (relay/ESP32) — needed to generate the setup QR code
    ap.add_argument("--home-wifi-ssid", default="", help="Home WiFi SSID to embed in the relay setup QR")
    ap.add_argument("--home-wifi-password", default="", help="Home WiFi password to embed in the relay setup QR")
    ap.add_argument("--home-server-host", default="", help="LAN IP the ESP32 should call back to (auto-detected if omitted)")
    args = ap.parse_args()

    tts.configure(engine=args.tts, piper_model=args.piper_model,
                  sapi_voice=args.sapi_voice, sapi_rate=args.sapi_rate)
    tts.init_tts()

    ocr.configure(engine=args.ocr_engine)
    print(f"[OCR] engine={args.ocr_engine}"
          + (" (falls back to easyocr automatically if it can't start)" if args.ocr_engine == "paddleocr" else ""))

    # Pre-load OCR model in a background thread so the first scan isn't slow
    import threading as _t
    _t.Thread(target=ocr.preload, daemon=True).start()

    money.configure(model_path=args.model, detect_conf=args.detect_conf,
                    box_conf=args.box_conf, reliable_conf=args.reliable_conf,
                    imgsz=args.imgsz, tta=(True if args.tta else None),
                    blur_radius=args.blur_radius)

    home.configure(wifi_ssid=args.home_wifi_ssid, wifi_password=args.home_wifi_password,
                   server_host=args.home_server_host)
    if not args.home_wifi_ssid:
        print("[HOME] --home-wifi-ssid not set — setup QR codes will have an empty SSID "
              "until you pass --home-wifi-ssid/--home-wifi-password.")
    # Start the home-automation dashboard/QR endpoint immediately at boot —
    # pairing is the FIRST thing you do with this feature, so the QR must be
    # servable before any "home" message has ever been received (unlike
    # lidar's dashboard, which reasonably waits for real data).
    home.start_http()

    print(f"[ROUTE] features: {', '.join(FEATURE_HANDLERS)}")
    print(f"[MONEY] model={money.MODEL_PATH}")
    print(f"[MONEY] detect_conf={money.YOLO_CONF_THRESHOLD} box_conf={money.BOX_CONF_MIN} "
          f"reliable_conf={money.RELIABLE_CONF} imgsz={money.YOLO_IMGSZ} tta={money.YOLO_TTA}")

    _stop = threading.Event()

    def _request_stop(*_):
        print("\n[NET] Shutdown requested — stopping…")
        _stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)
    srv.settimeout(1.0)          # wake up every second to check for stop signal
    print(f"[NET] Listening on {args.host}:{args.port}")
    tts.local_speak("Feature server ready.")

    try:
        while not _stop.is_set():
            try:
                conn, addr = srv.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue           # just re-check _stop
    finally:
        print("[NET] Server stopped.")
        srv.close()


if __name__ == "__main__":
    main()