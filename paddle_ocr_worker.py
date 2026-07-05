#!/usr/bin/env python3
"""
paddle_ocr_worker.py — isolated PaddleOCR inference process.

Runs PaddleOCR in its OWN Python process. handlers/ocr.py spawns this as a
subprocess on first use and talks to it over a localhost-only TCP socket,
using the exact same 4-byte length-prefixed JSON framing as protocol.py
(imported directly — it's stdlib-only, safe to share across processes).

Why a separate process at all: the original hypothesis was that
PaddlePaddle-GPU's own CUDA runtime conflicts with PyTorch's once `torch` is
imported in the same process (money.py's YOLO loader always does this).
Process isolation was built to eliminate that specific conflict. Direct
testing later proved that hypothesis incomplete on this deployment: `import
paddle` currently fails on Windows (OSError: WinError 127 loading
cusparse64_12.dll) identically with or without torch imported, in both GPU
and CPU device modes, and on two different paddlepaddle-gpu versions
(3.0.0rc1 and the current stable 3.3.1) — see handlers/ocr.py's module
docstring for the full diagnosis. This worker is kept anyway because it's
still the right design regardless: money.py's YOLO (PyTorch) and this
process get independent CUDA contexts, so a problem in one GPU stack can
never take the other down, and if the underlying paddle bug is ever fixed,
this process boundary costs nothing extra.

Protocol:
  request  {"cmd": "ping"}                      -> {"ok": true}
  request  {"cmd": "ocr", "image_b64": "<jpg>"}  -> {"ok": true, "detections": [[bbox, text, conf], ...]}
                                                  -> {"ok": false, "error": "..."}
"""

import argparse
import base64
import socket
import sys
import threading

import cv2
import numpy as np

from protocol import recv_msg, send_msg

# Force UTF-8 stdout/stderr regardless of the Windows console's default
# codepage — see the identical guard in feature_server.py for why.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_engine = None
_infer_lock = threading.Lock()


def _get_engine(device: str):
    global _engine
    if _engine is not None:
        return _engine
    from paddleocr import PaddleOCR
    print(f"[PADDLE-WORKER] Loading PaddleOCR (device: {device}) ...", flush=True)
    try:
        # PaddleOCR >=3.x constructor
        _engine = PaddleOCR(use_textline_orientation=True, lang="en", device=device)
    except TypeError:
        # PaddleOCR 2.x constructor (e.g. if paddlepaddle-gpu is ever downgraded
        # off the paddlex-based 3.x line to work around this deployment's
        # Windows DLL loading issue — see handlers/ocr.py's module docstring)
        print("[PADDLE-WORKER] 3.x constructor rejected; trying PaddleOCR 2.x API ...", flush=True)
        _engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=(device != "cpu"), show_log=False)
    print(f"[PADDLE-WORKER] Ready — device: {device}", flush=True)
    return _engine


def _run_ocr(engine, image_b64: str) -> list:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")

    out = []
    with _infer_lock:
        if hasattr(engine, "predict"):
            # PaddleOCR >=3.x pipeline API
            results = engine.predict(img)
            if not results:
                return []
            result = results[0]
            for bbox, text, conf in zip(result["rec_polys"], result["rec_texts"], result["rec_scores"]):
                text = str(text).strip()
                if text:
                    # numpy ints/arrays aren't JSON-serializable — coerce to plain Python
                    out.append([[[int(x), int(y)] for x, y in bbox], text, float(conf)])
        else:
            # PaddleOCR 2.x legacy API: .ocr() -> [[[box, (text, conf)], ...]]
            result = engine.ocr(img, cls=True)
            for item in (result[0] if result and result[0] else []):
                try:
                    bbox, (text, conf) = item
                    text = str(text).strip()
                    if text:
                        out.append([[[int(x), int(y)] for x, y in bbox], text, float(conf)])
                except (TypeError, ValueError, IndexError):
                    continue
    return out


def _handle_conn(conn: socket.socket, engine) -> None:
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            cmd = msg.get("cmd")
            if cmd == "ping":
                send_msg(conn, {"ok": True})
            elif cmd == "ocr":
                try:
                    detections = _run_ocr(engine, msg["image_b64"])
                    send_msg(conn, {"ok": True, "detections": detections})
                except Exception as e:
                    send_msg(conn, {"ok": False, "error": str(e)})
            else:
                send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"})
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--device", default="cpu", help='"cpu" or "gpu:0"')
    args = ap.parse_args()

    try:
        engine = _get_engine(args.device)
    except Exception as e:
        print(f"[PADDLE-WORKER] FATAL: could not load PaddleOCR ({e!r})", flush=True)
        sys.exit(1)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(8)
    print(f"[PADDLE-WORKER] Listening on 127.0.0.1:{args.port}", flush=True)

    while True:
        conn, _addr = srv.accept()
        threading.Thread(target=_handle_conn, args=(conn, engine), daemon=True).start()


if __name__ == "__main__":
    main()
