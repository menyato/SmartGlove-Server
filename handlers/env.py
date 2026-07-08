"""
handlers/env.py — server-side sink for the Environmental Awareness feature.

The env feature runs its own Gemini calls entirely on the OrangePi (no server
compute needed). This handler exists only so the key frames it sent to Gemini
and the description Gemini returned are archived on the laptop for review /
debugging. Nothing here is voiced back — replies carry an empty "tts".

Messages (feature="env"):
  action="scan"          — one completed scan.
      session   : str    session name (may be the silent auto name)
      user      : str    the prompt sent to Gemini
      reply     : str    Gemini's description
      frames_b64: [str]  base64 JPEG key frames (0..N)
      keyframes : int    how many frames were sent
  action="feature_state" — lifecycle ping (started / stopped / …); logged only.

Everything lands under handlers/env_scans/<session>/ :
  scan_<n>_<ts>_f<k>.jpg   the archived key frames
  log.jsonl                one line per scan (user, reply, frame filenames)
"""

import base64
import json
import os
import re
import threading
import time

from handlers.base import FeatureHandler

_BASE_DIR = os.path.join(os.path.dirname(__file__), "env_scans")
_lock = threading.Lock()


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", (name or "").strip())[:60] or "session"


class EnvHandler(FeatureHandler):
    name = "env"

    def __init__(self) -> None:
        # scan counter is per-session so filenames stay ordered across a run
        self._counts: dict[str, int] = {}

    def handle(self, msg: dict) -> dict:
        action = msg.get("action", "scan")
        if action == "feature_state":
            print(f"[ENV] state: {msg.get('feature_name','env')} "
                  f"-> {msg.get('state','?')}"
                  + (f" (session {msg.get('session')})" if msg.get("session") else ""))
            return {"tts": "", "quit": False}
        if action == "scan":
            return self._save_scan(msg)
        return {"tts": "", "quit": False}

    def _save_scan(self, msg: dict) -> dict:
        session = _safe(msg.get("session", "session"))
        sess_dir = os.path.join(_BASE_DIR, session)
        try:
            os.makedirs(sess_dir, exist_ok=True)
        except OSError as e:
            print(f"[ENV] Could not create {sess_dir}: {e}")
            return {"tts": "", "quit": False}

        with _lock:
            n = self._counts.get(session, 0) + 1
            self._counts[session] = n

        ts = time.strftime("%Y%m%d_%H%M%S")
        frame_files = []
        for k, b64 in enumerate(msg.get("frames_b64", []) or []):
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            fname = f"scan_{n}_{ts}_f{k}.jpg"
            try:
                with open(os.path.join(sess_dir, fname), "wb") as f:
                    f.write(data)
                frame_files.append(fname)
            except OSError as e:
                print(f"[ENV] Frame write error: {e}")

        rec = {
            "ts":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan":     n,
            "session":  msg.get("session", session),
            "user":     msg.get("user", ""),
            "reply":    msg.get("reply", ""),
            "keyframes": msg.get("keyframes", len(frame_files)),
            "frames":   frame_files,
        }
        try:
            with open(os.path.join(sess_dir, "log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[ENV] Log write error: {e}")

        print(f"[ENV] Archived scan {n} for '{session}': "
              f"{len(frame_files)} frame(s), {len(rec['reply'].split())} reply words "
              f"-> {sess_dir}")
        return {"tts": "", "quit": False}
