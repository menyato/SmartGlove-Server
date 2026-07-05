"""
handlers/ocr.py — Book Reader server handler.

Two selectable OCR engines, set via --ocr-engine on the CLI (configure()):

  "easyocr"   (default) — torch-based, shares PyTorch's CUDA runtime with
              Money/YOLO with zero conflict. Confirmed working on GPU.

  "paddleocr" — runs in an ISOLATED subprocess (paddle_ocr_worker.py), never
              importing torch, for accuracy reasons (PaddleOCR is generally
              considered more accurate for dense printed text than EasyOCR).
              On this deployment, paddle's own `import paddle` step currently
              fails on Windows (OSError: WinError 127 loading
              cusparse64_12.dll) in BOTH GPU and CPU device modes, confirmed
              by direct testing to be unrelated to PyTorch, to paddle version
              (reproduced on 3.0.0rc1 AND the current stable 3.3.1), and to
              missing VC++ redistributables — the isolated worker subprocess
              itself fails to start in this environment. Rather than block on
              that, this handler tries to start the PaddleOCR worker when
              "paddleocr" is selected, and transparently falls back to
              EasyOCR (loading it if needed) for that request — and every
              request after, without retrying the failed worker every time —
              if the worker can't come up. If the underlying environment issue
              is ever resolved (paddle upstream fix, or downgrading to the
              pre-paddlex 2.x line), "paddleocr" will simply start working
              with no further code changes.

Thread-safe via _ocr_lock / _worker_lock. Per-scan metrics saved to
ocr_metrics.jsonl. Debug images auto-rotated (newest MAX_DEBUG_FILES kept).
"""

import base64
import json
import os
import re
import socket as _socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from handlers.base import FeatureHandler
from protocol import recv_msg, send_msg

try:
    import tts as _tts
    _TTS_OK = True
except Exception:
    _TTS_OK = False

MAX_CHUNK_TTS  = 20
_CHUNK_TARGET  = 10
CONF_THRESHOLD = 0.30
MAX_DEBUG_FILES = 20

DEBUG_DIR    = os.path.join(os.path.dirname(__file__), "..", "ocr_debug")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "ocr_metrics.jsonl")
# Same consolidated file feature_server.py logs per-request latency/response
# time into (see Section 9 of the report) — this appends the one-time model
# load cost so it lives alongside every other feature's timing data.
_FEATURE_METRICS_PATH = os.path.join(os.path.dirname(__file__), "feature_metrics.jsonl")

os.makedirs(DEBUG_DIR, exist_ok=True)

# ── metrics ───────────────────────────────────────────────────────────────────

_metrics_lock = threading.Lock()


def _log_load_metric(ms: float, ok: bool, **extra) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "feature": "ocr",
           "type": "model_load", "processing_ms": round(ms, 1), "ok": ok, **extra}
    with _metrics_lock:
        try:
            with open(_FEATURE_METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def _log_metric(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _metrics_lock:
        try:
            with open(METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


# ── debug image rotation ──────────────────────────────────────────────────────

def _rotate_debug() -> None:
    """Delete oldest files so DEBUG_DIR never exceeds MAX_DEBUG_FILES entries."""
    try:
        entries = sorted(
            (e for e in os.scandir(DEBUG_DIR) if e.name.endswith(".jpg")),
            key=lambda e: e.stat().st_mtime,
        )
        for entry in entries[:-MAX_DEBUG_FILES]:
            try:
                os.unlink(entry.path)
            except OSError:
                pass
    except OSError:
        pass


# ── CONFIG (override via configure() from the CLI) ────────────────────────────
OCR_ENGINE = "easyocr"   # "easyocr" | "paddleocr"


def configure(engine: "str | None" = None) -> None:
    global OCR_ENGINE
    if engine is not None:
        OCR_ENGINE = engine


# ── EasyOCR singleton (default engine; always the fallback for "paddleocr") ───

_easyocr_engine = None
_easyocr_lock   = threading.Lock()


def _build_easyocr(gpu: bool):
    import easyocr
    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def _get_easyocr():
    global _easyocr_engine
    with _easyocr_lock:
        if _easyocr_engine is not None:
            return _easyocr_engine
        t0 = time.time()
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        label = "GPU" if use_gpu else "CPU"
        print(f"[OCR] Loading EasyOCR (device: {label}) ...")

        try:
            if use_gpu:
                try:
                    _easyocr_engine = _build_easyocr(gpu=True)
                except Exception as gpu_err:
                    # Belt-and-suspenders: EasyOCR shares torch's CUDA runtime
                    # so this shouldn't normally fail once torch itself
                    # reports CUDA available, but don't take OCR down over a
                    # GPU-stack problem (driver issue, OOM, etc.) — retry once
                    # on CPU rather than crash every scan request.
                    print(f"[OCR] GPU init failed ({gpu_err!r}); falling back to CPU.")
                    label = "CPU (GPU init failed)"
                    _easyocr_engine = _build_easyocr(gpu=False)
            else:
                _easyocr_engine = _build_easyocr(gpu=False)
        except Exception as e:
            _log_load_metric((time.time() - t0) * 1000, ok=False, device=f"easyocr:{label}", error=str(e))
            raise

        print(f"[OCR] EasyOCR ready — device: {label}")
        _log_load_metric((time.time() - t0) * 1000, ok=True, device=f"easyocr:{label}")
        return _easyocr_engine


# ── PaddleOCR isolated worker (opt-in via --ocr-engine paddleocr) ─────────────
# Runs in its own process (paddle_ocr_worker.py) so paddle's CUDA runtime never
# shares a process with PyTorch's. See the module docstring for why this
# currently falls back to EasyOCR on this deployment.

_WORKER_PORT = 9091
_worker_proc = None
_worker_lock = threading.Lock()
_worker_device_label = "?"
_paddle_available = True   # set False after one failed start; stop retrying every request


def _worker_script_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "paddle_ocr_worker.py")


def _worker_ping(timeout: float = 1.0) -> bool:
    try:
        with _socket.create_connection(("127.0.0.1", _WORKER_PORT), timeout=timeout) as s:
            send_msg(s, {"cmd": "ping"})
            resp = recv_msg(s)
            return bool(resp and resp.get("ok"))
    except OSError:
        return False


def _spawn_worker(device: str) -> bool:
    global _worker_proc, _worker_device_label
    script = _worker_script_path()
    args = [sys.executable, script, "--port", str(_WORKER_PORT), "--device", device]
    _worker_proc = subprocess.Popen(args, cwd=os.path.dirname(script))
    _worker_device_label = device

    deadline = time.time() + 90.0   # first load can be slow (model weights, GPU init)
    while time.time() < deadline:
        if _worker_proc.poll() is not None:
            return False   # crashed during startup
        if _worker_ping(timeout=0.5):
            return True
        time.sleep(0.5)
    return False


def _ensure_paddle_worker() -> bool:
    global _paddle_available
    with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None and _worker_ping():
            return True
        if not _paddle_available:
            return False   # already failed once this process lifetime — don't retry every request

        t0 = time.time()
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

        ok = _spawn_worker("gpu:0") if use_gpu else False
        if use_gpu and not ok:
            print("[OCR] PaddleOCR GPU worker failed to start; trying CPU worker...")
        if not ok:
            ok = _spawn_worker("cpu")

        if not ok:
            print("[OCR] PaddleOCR worker could not start on this machine — "
                  "falling back to EasyOCR for OCR requests.")
            _paddle_available = False
            _log_load_metric((time.time() - t0) * 1000, ok=False,
                             device="paddleocr", error="worker_failed_to_start")
            return False

        print(f"[OCR] PaddleOCR worker ready — device: {_worker_device_label}")
        _log_load_metric((time.time() - t0) * 1000, ok=True,
                         device=f"paddleocr:{_worker_device_label}")
        return True


def _run_paddle_ocr(img: np.ndarray) -> list:
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
    with _socket.create_connection(("127.0.0.1", _WORKER_PORT), timeout=30.0) as s:
        send_msg(s, {"cmd": "ocr", "image_b64": b64})
        resp = recv_msg(s)
    if not resp or not resp.get("ok"):
        raise RuntimeError(resp.get("error") if resp else "worker connection lost")

    out = []
    for bbox, text, conf in resp["detections"]:
        text = str(text).strip()
        if text:
            out.append((bbox, text, float(conf)))
    return out


def preload() -> None:
    """Call at server startup to warm up the configured engine (and EasyOCR,
    if PaddleOCR was requested but can't start — so the fallback is ready too)."""
    if OCR_ENGINE == "paddleocr" and _ensure_paddle_worker():
        return
    _get_easyocr()


# ── image helpers ─────────────────────────────────────────────────────────────

def _decode_jpeg(b64: str) -> "np.ndarray | None":
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _auto_rotate(img: np.ndarray) -> np.ndarray:
    """Rotate portrait images to landscape so text lines are horizontal."""
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    img = _auto_rotate(img)
    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


# ── OCR inference ─────────────────────────────────────────────────────────────

def _run_ocr(img: np.ndarray) -> list:
    """Return list of (bbox, text, conf), from whichever engine is configured.

    bbox is a 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in both
    engines' native output, matching the downstream top-to-bottom sort
    (`r[0][0][1]`) that already expects this shape.
    """
    if OCR_ENGINE == "paddleocr" and _ensure_paddle_worker():
        try:
            return _run_paddle_ocr(img)
        except Exception as e:
            print(f"[OCR] PaddleOCR worker request failed ({e}); "
                  "falling back to EasyOCR for this request.")

    engine = _get_easyocr()
    results = engine.readtext(img)
    out = []
    for bbox, text, conf in results:
        text = str(text).strip()
        if text:
            out.append((bbox, text, float(conf)))
    return out


# ── page number detection ─────────────────────────────────────────────────────

_PAGE_PATTERNS = [
    r'\bpage\s+(\d+)\b',
    r'\bpg\.?\s*(\d+)\b',
    r'^(\d+)$',
    r'\b(\d+)\s+of\s+\d+\b',
    r'[-–]\s*(\d+)\s*[-–]',
]


def _detect_page(lines: list) -> "int | None":
    """Search bottom quarter first (most common page number position)."""
    quarter = max(1, len(lines) // 4)
    ordered = lines[-quarter:] + lines
    for line in ordered:
        for pat in _PAGE_PATTERNS:
            m = re.search(pat, line.strip(), re.IGNORECASE)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 9999:
                        return n
                except (ValueError, IndexError):
                    pass
    return None


# ── text chunk helpers ────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list = []
    for s in sentences:
        words = s.split()
        if len(words) <= int(_CHUNK_TARGET * 1.5):
            if s.strip():
                chunks.append(s.strip())
        else:
            parts = re.split(r'[,;]\s*', s)
            buf: list = []
            for p in parts:
                buf.extend(p.split())
                if len(buf) >= _CHUNK_TARGET:
                    chunks.append(" ".join(buf))
                    buf = []
            if buf:
                chunks.append(" ".join(buf))
    return [c for c in chunks if c]


# ── TTS synthesis ─────────────────────────────────────────────────────────────

def _synth(text: str) -> "bytes | None":
    if not _TTS_OK:
        return None
    try:
        wav, engine = _tts.synth_wav(text)
        if wav:
            print(f"[OCR-TTS] {len(text)} chars → {len(wav)//1024} KB ({engine})")
        return wav
    except Exception as e:
        print(f"[OCR-TTS] synth error: {e}")
        return None


# ── handler ───────────────────────────────────────────────────────────────────

class OCRHandler(FeatureHandler):
    name = "ocr"

    def handle(self, msg: dict) -> dict:
        mtype = msg.get("type", "scan")
        if mtype == "hello":
            return {"tts": "", "text": "", "quit": False}
        if mtype == "scan":
            return self._handle_scan(msg)
        return {"tts": "", "text": "Unknown OCR command.", "quit": False}

    def _handle_scan(self, msg: dict) -> dict:
        t0  = time.time()
        b64 = msg.get("frame")
        if not b64:
            print("[OCR] ERROR: no 'frame' field in message")
            _log_metric({"error": "no_frame", "processing_ms": 0})
            return {"tts": "", "text": "No image received.", "quit": False}

        img = _decode_jpeg(b64)
        if img is None:
            print("[OCR] ERROR: could not decode JPEG")
            _log_metric({"error": "decode_failed", "processing_ms": 0})
            return {"tts": "", "text": "Could not decode image.", "quit": False}

        print(f"[OCR] Image decoded: {img.shape[1]}x{img.shape[0]} px")

        ts       = time.strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(DEBUG_DIR, f"scan_{ts}_raw.jpg")
        cv2.imwrite(raw_path, img)

        processed = _preprocess(img)
        proc_path = os.path.join(DEBUG_DIR, f"scan_{ts}_proc.jpg")
        cv2.imwrite(proc_path, processed)
        _rotate_debug()

        print(f"[OCR] Preprocessed: {processed.shape[1]}x{processed.shape[0]} px")

        try:
            raw = _run_ocr(processed)
        except Exception as e:
            print(f"[OCR] Inference error: {e}")
            _log_metric({"error": str(e),
                         "processing_ms": int((time.time() - t0) * 1000)})
            return {"tts": "", "text": f"OCR error: {e}", "quit": False}

        print(f"[OCR] Raw detections: {len(raw)}")
        for _, text, conf in raw:
            flag = "" if conf >= CONF_THRESHOLD else f"  ← below {CONF_THRESHOLD:.2f} (filtered)"
            print(f"[OCR]   conf={conf:.3f}  text={text!r}{flag}")

        if not raw:
            print("[OCR] No detections — image may be blank or unreadable")
            _log_metric({"words": 0, "lines": 0, "page": None, "conf_avg": 0.0,
                         "processing_ms": int((time.time() - t0) * 1000),
                         "error": "no_detections"})
            return {"tts": "", "text": "", "quit": False}

        # Sort top-to-bottom: use top-left corner y-coordinate
        raw.sort(key=lambda r: r[0][0][1])

        passing = [(text, conf) for _, text, conf in raw
                   if conf >= CONF_THRESHOLD]

        if not passing:
            print(f"[OCR] All {len(raw)} detections below threshold {CONF_THRESHOLD:.2f}")
            _log_metric({"words": 0, "lines": 0, "page": None, "conf_avg": 0.0,
                         "detections_total": len(raw), "detections_kept": 0,
                         "processing_ms": int((time.time() - t0) * 1000),
                         "error": "all_below_threshold"})
            return {"tts": "", "text": "", "quit": False}

        lines      = [t for t, _ in passing]
        conf_avg   = sum(c for _, c in passing) / len(passing)
        page       = _detect_page(lines)
        full_text  = " ".join(lines)
        word_count = len(full_text.split())
        proc_ms    = int((time.time() - t0) * 1000)

        print(f"[OCR] Page {page} | {len(lines)} lines | {word_count} words | "
              f"avg conf={conf_avg:.2f} | {proc_ms} ms")

        _log_metric({
            "words":             word_count,
            "lines":             len(lines),
            "page":              page,
            "conf_avg":          round(conf_avg, 3),
            "detections_total":  len(raw),
            "detections_kept":   len(passing),
            "processing_ms":     proc_ms,
            "error":             None,
        })

        announcement = (f"Page {page}. {word_count} words."
                        if page is not None else
                        f"No page number. {word_count} words.")

        chunks     = _chunk_text(full_text)
        ann_wav    = _synth(announcement)
        chunk_wavs = [_synth(c) for c in chunks[:MAX_CHUNK_TTS]]

        resp: dict = {
            "tts":        "",
            "text":       full_text,
            "page":       page,
            "word_count": word_count,
            "chunks":     chunks,
            "quit":       False,
        }
        if ann_wav:
            resp["announcement_wav"] = base64.b64encode(ann_wav).decode("ascii")
        if chunk_wavs and any(w for w in chunk_wavs):
            resp["chunk_wavs"] = [
                base64.b64encode(w).decode("ascii") if w else None
                for w in chunk_wavs
            ]
        return resp
