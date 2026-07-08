"""
handlers/env.py — server-side Gemini proxy + archive for Environmental Awareness.

Why this lives on the laptop: the OrangePi glove runs on a phone hotspot whose
DNS drops out intermittently ("Name or service not known"), so calling Gemini
from the Pi is unreliable. The laptop has stable internet, so the glove sends
the key frames + prompt here, the laptop calls Gemini, and the reply goes back
over the LAN socket. Every scan's frames + reply are archived under
handlers/env_scans/<session>/ so the results can be reviewed for accuracy.

The glove keeps its own on-Pi Gemini client as an automatic fallback, so if the
laptop can't reach Gemini either, it still tries directly.

Messages (feature="env"):
  action="selftest"      -> {"gemini_ready": bool}
  action="ask"           -> run Gemini, archive, return the reply
      session     : str
      system      : str    full system_instruction (prompt + convo history)
      user        : str    the user's text this turn
      frames_b64  : [str]  base64 JPEG key frames (may be empty)
      keyframes   : int
      save_images : bool   archive the frames as files (True for scans)
    reply -> {"ok": bool, "reply": str, "error": str, "tts": "", "quit": False}
  action="scan"          -> archive-only (used when the Pi answered locally)
  action="feature_state" -> lifecycle ping; logged only

Setup on the laptop (server_app):
    pip install google-genai
    set GEMINI_API_KEY=your_key         (or add it to ~/.env / ~/keys.env,
                                         or paste into GEMINI_API_KEY_HARDCODED)
"""

import base64
import json
import os
import re
import threading
import time

from handlers.base import FeatureHandler

try:
    from google import genai
    from google.genai import types as _gtypes
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "env_scans"))
_lock = threading.Lock()
try:
    os.makedirs(_BASE_DIR, exist_ok=True)
except OSError:
    pass
print(f"[ENV] Scan archive dir: {_BASE_DIR}"
      + ("" if _GENAI_OK else "  (google-genai NOT installed — server can't run Gemini)"))

GEMINI_MODEL = "gemini-2.5-flash"
_KEY_ENV     = "GEMINI_API_KEY"

# Optional: paste your key here if you don't want to set an env var / file on
# the laptop. Leave "" to use GEMINI_API_KEY / ~/.env / ~/keys.env instead.
GEMINI_API_KEY_HARDCODED = ""


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", (name or "").strip())[:60] or "session"


def _load_api_key() -> str:
    if GEMINI_API_KEY_HARDCODED.strip():
        return GEMINI_API_KEY_HARDCODED.strip()
    key = os.environ.get(_KEY_ENV, "").strip()
    if key:
        return key
    for path in (os.path.expanduser("~/.env"), os.path.expanduser("~/keys.env")):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(_KEY_ENV + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


# One client for the whole process (created lazily on first use).
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if not _GENAI_OK:
        return None
    with _client_lock:
        if _client is not None:
            return _client
        key = _load_api_key()
        if not key:
            print("[ENV] No Gemini key on the laptop (GEMINI_API_KEY / ~/.env / "
                  "~/keys.env / GEMINI_API_KEY_HARDCODED all empty).")
            return None
        try:
            _client = genai.Client(api_key=key)
            print("[ENV] Server Gemini client ready.")
        except Exception as e:
            print(f"[ENV] Server Gemini init failed: {e}")
            _client = None
        return _client


def _is_transient_net_err(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return any(k in s for k in (
        "connecterror", "connecttimeout", "readtimeout", "read timeout",
        "name or service not known", "getaddrinfo", "errno -2", "errno -3",
        "connection reset", "connection aborted", "remoteprotocol",
        "timed out", "network is unreachable",
    ))


def _run_gemini(client, system: str, user_text: str, jpegs: list) -> str:
    contents = []
    for i, jpg in enumerate(jpegs):
        contents.append(f"[Frame {i + 1} of {len(jpegs)}]")
        contents.append(_gtypes.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
    contents.append(f"User: {user_text}")

    cfg_kwargs = dict(system_instruction=system, max_output_tokens=1200,
                      temperature=0.4)
    try:
        cfg_kwargs["thinking_config"] = _gtypes.ThinkingConfig(thinking_budget=0)
    except (AttributeError, TypeError):
        pass
    config = _gtypes.GenerateContentConfig(**cfg_kwargs)

    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=config,
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            if attempt < 3 and _is_transient_net_err(e):
                print(f"[ENV] Server Gemini blip ({attempt}/3): {e} — retrying")
                time.sleep(1.5)
                continue
            raise
    raise last_err


class EnvHandler(FeatureHandler):
    name = "env"

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def handle(self, msg: dict) -> dict:
        action = msg.get("action", "scan")
        if action == "selftest":
            ready = _get_client() is not None
            print(f"[ENV] selftest -> gemini_ready={ready}")
            return {"gemini_ready": ready, "tts": "", "quit": False}
        if action == "ask":
            return self._ask(msg)
        if action == "scan":
            self._save_scan(msg, reply=msg.get("reply", ""))
            return {"tts": "", "quit": False}
        if action == "feature_state":
            print(f"[ENV] state: {msg.get('feature_name','env')} "
                  f"-> {msg.get('state','?')}"
                  + (f" (session {msg.get('session')})" if msg.get("session") else ""))
            return {"tts": "", "quit": False}
        return {"tts": "", "quit": False}

    # ── Gemini ────────────────────────────────────────────────────────────────
    def _ask(self, msg: dict) -> dict:
        # BACKUP Gemini path only — no archiving here. The glove always sends a
        # separate "scan" message that saves the frames + reply for reference,
        # regardless of which side ran Gemini.
        client = _get_client()
        if client is None:
            return {"ok": False, "reply": "",
                    "error": "server has no Gemini client (missing package or key)",
                    "tts": "", "quit": False}
        jpegs = []
        for b64 in msg.get("frames_b64", []) or []:
            try:
                jpegs.append(base64.b64decode(b64))
            except (ValueError, TypeError):
                pass
        system = msg.get("system", "") or ""
        user   = msg.get("user", "") or ""
        try:
            reply = _run_gemini(client, system, user, jpegs)
        except Exception as e:
            print(f"[ENV] Backup Gemini call failed: {e}")
            return {"ok": False, "reply": "", "error": str(e),
                    "tts": "", "quit": False}
        print(f"[ENV] backup ask -> {len(reply.split())} words")
        # No "tts": the glove speaks the reply itself with local espeak; setting
        # tts would make the laptop synthesize and speak it here instead.
        return {"ok": True, "reply": reply, "error": "", "tts": "", "quit": False}

    # ── archive ─────────────────────────────────────────────────────────────
    def _save_scan(self, msg: dict, reply: str = "", jpegs: "list | None" = None) -> None:
        session  = _safe(msg.get("session", "session"))
        sess_dir = os.path.join(_BASE_DIR, session)
        try:
            os.makedirs(sess_dir, exist_ok=True)
        except OSError as e:
            print(f"[ENV] Could not create {sess_dir}: {e}")
            return

        with _lock:
            n = self._counts.get(session, 0) + 1
            self._counts[session] = n

        # Decode frames if not already decoded (the "scan" archive path passes
        # base64 in the message; the "ask" path passes decoded bytes).
        if jpegs is None:
            jpegs = []
            for b64 in msg.get("frames_b64", []) or []:
                try:
                    jpegs.append(base64.b64decode(b64))
                except (ValueError, TypeError):
                    pass

        ts = time.strftime("%Y%m%d_%H%M%S")
        frame_files = []
        if msg.get("save_images", True):
            for k, data in enumerate(jpegs):
                fname = f"scan_{n}_{ts}_f{k}.jpg"
                try:
                    with open(os.path.join(sess_dir, fname), "wb") as f:
                        f.write(data)
                    frame_files.append(fname)
                except OSError as e:
                    print(f"[ENV] Frame write error: {e}")

        rec = {
            "ts":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan":      n,
            "session":   msg.get("session", session),
            "user":      msg.get("user", ""),
            "reply":     reply,
            "keyframes": msg.get("keyframes", len(jpegs)),
            "frames":    frame_files,
        }
        try:
            with open(os.path.join(sess_dir, "log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[ENV] Log write error: {e}")

        print(f"[ENV] Archived #{n} for '{session}': {len(frame_files)} frame(s), "
              f"{len(reply.split())} reply words -> {sess_dir}")
