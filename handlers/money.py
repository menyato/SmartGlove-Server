"""
handlers/money.py — the MONEY feature, self-contained.

Everything currency-specific from the original server.py lives here: YOLO
detection, the per-session voice state machine (ASK_CURRENCY → ASK_AMOUNT →
CONFIRM_AMOUNT → SCANNING), change-making, and the confirmation logic. The
general server (feature_server.py) just routes "money" messages to
MoneyHandler.handle() and voices the reply.

The YOLO model loads lazily on the first scan, so the server can start (and
serve other features) even before a scan happens, and the model path can be
set from the CLI via configure().
"""

import base64
import json
import os
import re
import threading
import time

import cv2
import numpy as np

from handlers.base import FeatureHandler

# ── CONFIG (override via configure() from the CLI) ────────────────────────────
MODEL_PATH = r"C:\Users\user\Desktop\Smart_Glove\runs\detect\runs\money_detection\yolov8l_currency-2\weights\best.pt"
USD_TO_LBP = 89000

YOLO_CONF_THRESHOLD = 0.30
BOX_CONF_MIN        = 0.40
RELIABLE_CONF       = 0.0     # 0 = off; >0 (e.g. 0.55) enables "please rescan" warnings
YOLO_IMGSZ          = 640
YOLO_TTA            = False
OVERLAP_RATIO_MAX   = 0.70
TOLERANCE_RATIO     = 0.005

SAVE_BLUR_RADIUS = 21
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR     = os.path.join(SCRIPT_DIR, "captured_scans")
os.makedirs(SAVE_DIR, exist_ok=True)
METRICS_PATH = os.path.join(SCRIPT_DIR, "money_metrics.jsonl")
# Same consolidated file feature_server.py logs per-request latency/response
# time into (see Section 9 of the report) — this appends the one-time model
# load cost so it lives alongside every other feature's timing data.
_FEATURE_METRICS_PATH = os.path.join(SCRIPT_DIR, "feature_metrics.jsonl")

# ── TRIGGER WORDS ─────────────────────────────────────────────────────────────
SCAN_TRIGGERS = {
    "scan", "capture", "snap", "shoot", "photo", "picture", "frame",
    "take picture", "take photo", "scan now", "do scan", "scan it", "yalla",
}
REDO_TRIGGERS = {"redo", "rescan", "scan again", "redo scan", "do over"}
DISCARD_TRIGGERS = {"discard", "cancel", "remove", "delete", "scratch",
                    "throw away", "forget it"}
CONFIRM_TRIGGERS = {"yes", "yeah", "yep", "yup", "correct", "confirm",
                    "confirmed", "right", "sure", "okay", "ok"}
REJECT_TRIGGERS  = {"no", "nope", "nah", "wrong", "incorrect", "change"}
QUIT_TRIGGERS = {"quit", "exit", "stop", "done", "finish", "bye", "goodbye"}
CURRENCY_USD  = {"dollar", "dollars", "usd", "buck", "bucks"}
CURRENCY_LBP  = {"lebanese", "lbp", "pound", "pounds", "lira", "lirah"}

# ── lazy YOLO + metrics ───────────────────────────────────────────────────────
_yolo_model  = None
_yolo_device = "cpu"
_yolo_lock = threading.Lock()
_metrics_lock = threading.Lock()


def configure(model_path: str | None = None, detect_conf: float | None = None,
              box_conf: float | None = None, reliable_conf: float | None = None,
              imgsz: int | None = None, tta: bool | None = None,
              blur_radius: int | None = None) -> None:
    global MODEL_PATH, YOLO_CONF_THRESHOLD, BOX_CONF_MIN, RELIABLE_CONF
    global YOLO_IMGSZ, YOLO_TTA, SAVE_BLUR_RADIUS
    if model_path is not None:
        MODEL_PATH = model_path
    if detect_conf is not None:
        YOLO_CONF_THRESHOLD = detect_conf
    if box_conf is not None:
        BOX_CONF_MIN = box_conf
    if reliable_conf is not None:
        RELIABLE_CONF = reliable_conf
    if imgsz is not None:
        YOLO_IMGSZ = imgsz
    if tta is not None:
        YOLO_TTA = tta
    if blur_radius is not None:
        SAVE_BLUR_RADIUS = blur_radius


def _log_load_metric(ms: float, ok: bool, **extra) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "feature": "money",
           "type": "model_load", "processing_ms": round(ms, 1), "ok": ok, **extra}
    with _metrics_lock:
        try:
            with open(_FEATURE_METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def _get_model():
    global _yolo_model, _yolo_device
    with _yolo_lock:
        if _yolo_model is None:
            t0 = time.time()
            try:
                import torch
                _yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                _yolo_device = "cpu"
            print(f"[YOLO] Loading model (device: {_yolo_device}) ...")
            try:
                from ultralytics import YOLO
                _yolo_model = YOLO(MODEL_PATH)
            except Exception as e:
                _log_load_metric((time.time() - t0) * 1000, ok=False,
                                 device=_yolo_device, error=str(e))
                raise
            print(f"[YOLO] ready — device: {_yolo_device}")
            _log_load_metric((time.time() - t0) * 1000, ok=True, device=_yolo_device)
    return _yolo_model


def log_metric(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _metrics_lock:
        try:
            with open(METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


# ── FRAME HELPERS ─────────────────────────────────────────────────────────────
def decode_jpeg(b64_str: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[CAM] cv2.imdecode returned None — bad JPEG?")
        return frame
    except Exception as e:
        print(f"[CAM] JPEG decode error: {e}")
        return None


def _blur_frame(frame: np.ndarray, radius: int) -> np.ndarray:
    k = radius if radius % 2 == 1 else radius + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def dump_frames(raw_frame: np.ndarray, annotated_frame: np.ndarray, scan_number: int) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"scan_{scan_number}_{ts}"
    annotated_path = os.path.join(SAVE_DIR, f"{base}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_frame)
    print(f"[DUMP] Annotated → {annotated_path}")
    if SAVE_BLUR_RADIUS > 0:
        blurred = _blur_frame(raw_frame, SAVE_BLUR_RADIUS)
        blur_path = os.path.join(SAVE_DIR, f"{base}_blurred.jpg")
        cv2.imwrite(blur_path, blurred)
        print(f"[DUMP] Blurred   → {blur_path}")
    try:
        cv2.imshow("Currency Detection", annotated_frame)
        cv2.waitKey(1)
    except Exception:
        pass   # headless server — no display


# ── CURRENCY HELPERS ──────────────────────────────────────────────────────────
def exact_tolerance(target_usd: float) -> float:
    return max(1000 / USD_TO_LBP, min(0.50, target_usd * TOLERANCE_RATIO))


def bill_label(name: str) -> str:
    parts = name.split("_")
    return f"{parts[1]} {parts[0].upper()}" if len(parts) == 2 else name


def build_bill_speech(bill_names: list) -> str:
    counts = {}
    for name in bill_names:
        label = bill_label(name)
        counts[label] = counts.get(label, 0) + 1
    parts = [f"one {l}" if q == 1 else f"{q} {l}" for l, q in counts.items()]
    return ", ".join(parts) if parts else "No bills detected."


def filter_overlapping_boxes(boxes: list) -> list:
    if len(boxes) <= 1:
        return boxes
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            xi1 = max(boxes[i]["x1"], boxes[j]["x1"])
            yi1 = max(boxes[i]["y1"], boxes[j]["y1"])
            xi2 = min(boxes[i]["x2"], boxes[j]["x2"])
            yi2 = min(boxes[i]["y2"], boxes[j]["y2"])
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            inter  = (xi2 - xi1) * (yi2 - yi1)
            area_j = ((boxes[j]["x2"] - boxes[j]["x1"]) *
                      (boxes[j]["y2"] - boxes[j]["y1"]))
            if area_j > 0 and inter / area_j > OVERLAP_RATIO_MAX:
                if boxes[i]["conf"] >= boxes[j]["conf"]:
                    keep[j] = False
                else:
                    keep[i] = False
    return [b for b, k in zip(boxes, keep) if k]


def suggest_return_bills(scanned_bills: list, extra_usd: float) -> list:
    tol          = exact_tolerance(extra_usd)
    sorted_bills = sorted(scanned_bills, key=lambda b: b["x_center"], reverse=True)
    for bill in sorted_bills:
        if abs(bill["value_usd"] - extra_usd) <= tol:
            return [bill]
    to_return = []
    remaining = extra_usd
    for bill in sorted_bills:
        if bill["value_usd"] <= remaining + tol:
            to_return.append(bill)
            remaining -= bill["value_usd"]
            if abs(remaining) <= tol:
                break
    return to_return


def speak_return_suggestion(bills: list) -> str:
    if not bills:
        return ""
    if len(bills) == 1:
        return f"Return the {bill_label(bills[0]['name'])} bill on the right."
    labels = [bill_label(b["name"]) for b in bills]
    return (f"Return these bills from right to left: "
            f"{', '.join(labels[:-1])} and {labels[-1]}.")


def parse_number(text: str, currency: str = "USD") -> int:
    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1_000_000,
    }
    text   = re.sub(r'(\d+)\.(\d+)', lambda m: m.group(1), text)
    text   = text.replace(",", "").replace("$", "").replace("£", "")
    tokens = text.replace("-", " ").split()
    current = total = 0
    for t in tokens:
        if t.isdigit():
            n = int(t)
            if n >= 1000:
                total += n
                current = 0
            else:
                current += n
        elif t in words:
            v = words[t]
            if v in (100, 1000, 1_000_000):
                current = max(current, 1) * v
                if v >= 1000:
                    total += current
                    current = 0
            else:
                current += v
    total += current
    if currency == "LBP" and 0 < total < 1000:
        total *= 1000
    return total


# ── SESSION STATE ─────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.state           = "ASK_CURRENCY"
        self.currency        = None
        self.target_usd      = 0.0
        self.pending_amount  = 0
        self.total_usd       = 0.0
        self.total_lbp       = 0.0
        self.all_bills: list = []
        self.last_scan_bills: list = []
        self.awaiting_return = False
        self.return_target   = 0.0
        self.scan_number     = 1
        self.undo_stack: list = []


def _snapshot(session: Session) -> dict:
    return {
        "total_usd":       session.total_usd,
        "total_lbp":       session.total_lbp,
        "all_bills":       list(session.all_bills),
        "last_scan_bills": list(session.last_scan_bills),
        "awaiting_return": session.awaiting_return,
        "return_target":   session.return_target,
        "scan_number":     session.scan_number,
    }


def _restore(session: Session, snap: dict) -> None:
    session.total_usd       = snap["total_usd"]
    session.total_lbp       = snap["total_lbp"]
    session.all_bills       = snap["all_bills"]
    session.last_scan_bills = snap["last_scan_bills"]
    session.awaiting_return = snap["awaiting_return"]
    session.return_target   = snap["return_target"]
    session.scan_number     = snap["scan_number"]


def _undo_last_scan(session: Session) -> bool:
    if not session.undo_stack:
        return False
    _restore(session, session.undo_stack.pop())
    return True


def _reset_session_totals(session: Session) -> None:
    session.total_usd       = 0.0
    session.total_lbp       = 0.0
    session.all_bills       = []
    session.last_scan_bills = []
    session.awaiting_return = False
    session.return_target   = 0.0
    session.undo_stack      = []


def _progress_after_discard(session: Session) -> str:
    equiv     = session.total_usd + (session.total_lbp / USD_TO_LBP)
    remaining = session.target_usd - equiv
    tol       = exact_tolerance(session.target_usd)
    s = f"Last scan discarded. Running total is now {equiv:.2f} dollars. "
    if remaining > tol:
        s += (f"Still need {remaining:.2f} dollars, "
              f"or {int(remaining*USD_TO_LBP):,} Lebanese pounds. "
              f"Say scan when ready.")
    elif abs(remaining) <= tol:
        s += "That matches the target. Say scan to confirm, or quit."
    else:
        s += f"That is {abs(remaining):.2f} dollars over the target."
    return s


# ── YOLO ANALYSIS ─────────────────────────────────────────────────────────────
def analyze_frame(frame: np.ndarray, session: Session) -> str:
    snap  = _snapshot(session)   # capture before any mutation; pushed only on success
    model = _get_model()
    t0 = time.time()
    results = model.predict(
        source=frame, conf=YOLO_CONF_THRESHOLD,
        iou=0.5, imgsz=YOLO_IMGSZ, augment=YOLO_TTA,
        verbose=False, stream=True, device=_yolo_device,
    )
    tolerance      = exact_tolerance(session.target_usd)
    prev_positions = {b["pos_id"] for b in session.last_scan_bills}

    this_scan_bills, new_bills, annotated = [], [], None
    all_confs = []
    raw_box_count = 0

    for r in results:
        raw_boxes = []
        for box in r.boxes:
            conf = float(box.conf[0])
            raw_box_count += 1
            if conf < BOX_CONF_MIN:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_boxes.append({
                "name":     r.names[int(box.cls[0])],
                "conf":     conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x_center": (x1 + x2) // 2,
                "pos_id":   f"{r.names[int(box.cls[0])]}_{x1//30}_{y1//30}",
            })

        clean     = filter_overlapping_boxes(raw_boxes)
        annotated = r.plot()

        for b in clean:
            parts = b["name"].split("_")
            try:
                denom = int(parts[1])
            except (IndexError, ValueError):
                continue
            if b["name"].startswith("usd"):
                value_usd = float(denom)
            elif b["name"].startswith("lbp"):
                value_usd = denom / USD_TO_LBP
            else:
                continue
            all_confs.append(b["conf"])
            bill = {"name": b["name"], "pos_id": b["pos_id"],
                    "x_center": b["x_center"], "value_usd": value_usd,
                    "conf": b["conf"]}
            this_scan_bills.append(bill)
            if b["pos_id"] not in prev_positions:
                new_bills.append(bill)

    infer_ms = (time.time() - t0) * 1000.0

    if annotated is not None:
        dump_frames(frame, annotated, session.scan_number)

    log_metric({
        "event": "yolo_scan",
        "scan_number": session.scan_number,
        "infer_ms": round(infer_ms, 1),
        "frame_shape": list(frame.shape),
        "raw_boxes": raw_box_count,
        "kept_bills": len(this_scan_bills),
        "new_bills": len(new_bills),
        "mean_conf": round(float(np.mean(all_confs)), 3) if all_confs else None,
        "min_conf":  round(float(np.min(all_confs)), 3) if all_confs else None,
        "low_conf_bills": sum(1 for b in this_scan_bills if b["conf"] < RELIABLE_CONF),
    })
    print(f"[YOLO] inference {infer_ms:.0f} ms | raw {raw_box_count} | "
          f"kept {len(this_scan_bills)} | new {len(new_bills)}"
          + (f" | conf {np.mean(all_confs):.2f}" if all_confs else ""))

    if not new_bills:
        session.last_scan_bills = this_scan_bills  # update tracking even on no-new
        if this_scan_bills:
            return ("Same bills detected as before. "
                    "Add new bills and say scan again.")
        return "No bills detected. Please try again."

    # Commit mutations; push undo snapshot only after we know changes happened
    scan_usd = sum(b["value_usd"] for b in new_bills if b["name"].startswith("usd"))
    scan_lbp = sum(b["value_usd"] * USD_TO_LBP for b in new_bills if b["name"].startswith("lbp"))

    session.undo_stack.append(snap)
    session.last_scan_bills = this_scan_bills
    session.total_usd += scan_usd
    session.total_lbp += scan_lbp
    session.all_bills.extend(new_bills)

    total_equiv   = session.total_usd + (session.total_lbp / USD_TO_LBP)
    remaining_usd = session.target_usd - total_equiv

    print(f"\n{'='*50}")
    print(f"SCAN #{session.scan_number}  —  {len(new_bills)} new bill(s)")
    print(f"  New this scan: USD ${scan_usd:.2f}  LBP {scan_lbp:,.0f}")
    print(f"  Running total: USD ${session.total_usd:.2f}  LBP {session.total_lbp:,.0f}")
    print(f"  Combined equiv: ${total_equiv:.2f}  |  Remaining: ${remaining_usd:.4f}")
    print("=" * 50)

    speech = f"Scan {session.scan_number}. {build_bill_speech([b['name'] for b in new_bills])}. "

    uncertain = [b for b in new_bills if b["conf"] < RELIABLE_CONF]
    if uncertain:
        speech += (f"Warning: {len(uncertain)} bill"
                   f"{'s were' if len(uncertain) > 1 else ' was'} not read "
                   f"clearly. Please flatten and rescan to be sure. ")

    if session.awaiting_return:
        returned_equiv = sum(b["value_usd"] for b in new_bills)
        diff           = session.return_target - returned_equiv
        ret_tol        = exact_tolerance(session.return_target)
        if abs(diff) <= ret_tol:
            speech += "Return confirmed. Transaction complete. Ready for next customer."
            _reset_session_totals(session)
        elif diff > 0:
            speech += (f"Still need to return {diff:.2f} dollars, "
                       f"or {int(diff*USD_TO_LBP):,} Lebanese pounds. ")
            speech += speak_return_suggestion(
                suggest_return_bills(session.all_bills, diff))
        else:
            speech += "You returned too much."
        session.scan_number += 1
        return speech

    if abs(remaining_usd) <= tolerance:
        speech += "Exact amount received. Transaction complete. Ready for next customer."
        _reset_session_totals(session)
    elif remaining_usd > 0:
        needed_lbp = int(remaining_usd * USD_TO_LBP)
        speech    += (f"Total received so far: {total_equiv:.2f} dollars. "
                      f"Still need {remaining_usd:.2f} dollars, "
                      f"or {needed_lbp:,} Lebanese pounds. "
                      f"Add more bills and say scan.")
    else:
        extra_usd               = abs(remaining_usd)
        session.awaiting_return = True
        session.return_target   = extra_usd
        speech += (f"Customer paid too much. "
                   f"Total is {total_equiv:.2f} dollars. "
                   f"Change is {extra_usd:.2f} dollars, "
                   f"or {int(extra_usd*USD_TO_LBP):,} Lebanese pounds. ")
        speech += speak_return_suggestion(
            suggest_return_bills(session.all_bills, extra_usd))
        speech += " Scan the returned money to confirm."

    session.scan_number += 1
    return speech


# ── VOICE STATE MACHINE ───────────────────────────────────────────────────────
def handle_voice(text: str, frame_b64: str | None, session: Session) -> dict:
    resp = {"tts": "", "quit": False}

    if any(w in text for w in QUIT_TRIGGERS):
        resp["tts"], resp["quit"] = "Goodbye.", True
        return resp

    if session.state == "ASK_CURRENCY":
        if any(w in text for w in CURRENCY_USD):
            session.currency = "USD"
            session.state    = "ASK_AMOUNT"
            resp["tts"]      = "US Dollar selected. How much in dollars?"
        elif any(w in text for w in CURRENCY_LBP):
            session.currency = "LBP"
            session.state    = "ASK_AMOUNT"
            resp["tts"]      = "Lebanese pound selected. How much in Lebanese pounds?"
        else:
            resp["tts"] = "Which currency? Say dollar or Lebanese pound."
        return resp

    if session.state == "ASK_AMOUNT":
        amount = parse_number(text, session.currency)
        if amount > 0:
            session.pending_amount = amount
            session.state = "CONFIRM_AMOUNT"
            if session.currency == "LBP":
                resp["tts"] = (f"You said {int(amount):,} Lebanese pounds. "
                               f"Say yes to confirm, or no to change.")
            else:
                resp["tts"] = (f"You said {amount} dollars. "
                               f"Say yes to confirm, or no to change.")
        else:
            resp["tts"] = f"Could not understand the amount. How much in {session.currency}?"
        return resp

    if session.state == "CONFIRM_AMOUNT":
        if any(w in text for w in REJECT_TRIGGERS):
            session.state = "ASK_AMOUNT"
            session.pending_amount = 0
            resp["tts"] = f"Okay. How much in {session.currency}?"
        elif any(w in text for w in CONFIRM_TRIGGERS):
            amt = session.pending_amount
            if session.currency == "LBP":
                session.target_usd = amt / USD_TO_LBP
                resp["tts"] = (f"Confirmed. Target is {int(amt):,} Lebanese pounds. "
                               f"Say scan when ready.")
            else:
                session.target_usd = float(amt)
                resp["tts"] = (f"Confirmed. Target is {session.target_usd:.2f} dollars. "
                               f"Say scan when ready.")
            session.state = "SCANNING"
        else:
            new_amount = parse_number(text, session.currency)
            if new_amount > 0:
                session.pending_amount = new_amount
                if session.currency == "LBP":
                    resp["tts"] = (f"You said {int(new_amount):,} Lebanese pounds. "
                                   f"Say yes to confirm, or no to change.")
                else:
                    resp["tts"] = (f"You said {new_amount} dollars. "
                                   f"Say yes to confirm, or no to change.")
            else:
                resp["tts"] = "Say yes to confirm, or no to change the amount."
        return resp

    if session.state == "SCANNING":
        if any(w in text for w in DISCARD_TRIGGERS):
            if _undo_last_scan(session):
                resp["tts"] = _progress_after_discard(session)
            else:
                resp["tts"] = "There is no scan to discard yet."
            return resp

        is_redo = any(w in text for w in REDO_TRIGGERS)
        is_scan = any(w in text for w in SCAN_TRIGGERS)
        if is_redo or is_scan:
            if frame_b64 is None:
                resp["tts"] = ("No frame received from camera. "
                               "Please point at the bills and try again.")
                return resp
            frame = decode_jpeg(frame_b64)
            if frame is None:
                resp["tts"] = ("Could not decode the camera image. "
                               "Check the webcam connection and try again.")
                return resp

            prefix = ""
            if is_redo:
                prefix = ("Redoing the last scan. " if _undo_last_scan(session) else "")
            print(f"[CAM] Received frame: {frame.shape}")
            resp["tts"] = prefix + analyze_frame(frame, session)
            return resp

        resp["tts"] = ("Say scan to capture, redo to scan again, "
                       "discard to remove the last scan, or quit to exit.")
        return resp

    resp["tts"] = "Say scan to capture, or quit to exit."
    return resp


# ── HANDLER (what the general server talks to) ────────────────────────────────
class MoneyHandler(FeatureHandler):
    name = "money"

    def __init__(self):
        self.session = Session()

    def handle(self, msg: dict) -> dict:
        mtype = msg.get("type", "")
        if mtype == "hello":
            return {"tts": "Connected. Which currency? Say dollar or Lebanese pound.",
                    "quit": False}
        if mtype == "voice":
            text = msg.get("text", "").lower()
            frame_b64 = msg.get("frame")
            return handle_voice(text, frame_b64, self.session)
        return {"tts": "Unknown message type.", "quit": False}