"""
tts.py — shared text-to-speech for the server (used by every feature).

Renders a reply string to a natural-voice WAV: Piper (neural, cross-platform)
preferred, Windows SAPI as fallback. This is infrastructure the general server
uses to voice ANY feature's reply — it is not money-specific.

Lifted from the original server.py so the voice is identical.
"""

import os
import shutil
import subprocess
import tempfile
import threading

# ── config (set via configure() before serving) ──────────────────────────────
TTS_ENGINE = "auto"      # auto | piper | sapi
PIPER_MODEL = ""         # path to a Piper .onnx voice; empty = Piper disabled
SAPI_VOICE = "Zira"      # preferred SAPI voice substring (Zira/David/Hazel/Aria)
SAPI_RATE = 0            # -10..10 ; 0 = default speed

_sapi_lock = threading.Lock()   # SAPI render is not thread-safe across clients


def configure(engine: str | None = None, piper_model: str | None = None,
              sapi_voice: str | None = None, sapi_rate: int | None = None) -> None:
    global TTS_ENGINE, PIPER_MODEL, SAPI_VOICE, SAPI_RATE
    if engine is not None:
        TTS_ENGINE = engine
    if piper_model is not None:
        PIPER_MODEL = piper_model
    if sapi_voice is not None:
        SAPI_VOICE = sapi_voice
    if sapi_rate is not None:
        SAPI_RATE = sapi_rate


# ── SAPI setup (Windows) ──────────────────────────────────────────────────────
_sapi_available = False
try:
    import win32com.client
    _speaker = win32com.client.Dispatch("SAPI.SpVoice")

    def _select_sapi_voice(substr: str):
        """Pick the SAPI voice whose name contains `substr` (case-insensitive)."""
        try:
            voices = _speaker.GetVoices()
            names = []
            for i in range(voices.Count):
                tok = voices.Item(i)
                desc = tok.GetDescription()
                names.append(desc)
                if substr and substr.lower() in desc.lower():
                    _speaker.Voice = tok
                    print(f"[TTS] SAPI voice → {desc}")
                    return
            print(f"[TTS] SAPI voices available: {names}")
            print(f"[TTS] (kept default voice; '{substr}' not found)")
        except Exception as e:
            print(f"[TTS] SAPI voice select failed: {e}")

    def sapi_to_wav(text: str) -> bytes | None:
        """Render SAPI speech to a real .wav via SpFileStream and return its bytes."""
        with _sapi_lock:
            tmp = None
            try:
                tmp = tempfile.mktemp(suffix=".wav")
                fs = win32com.client.Dispatch("SAPI.SpFileStream")
                fs.Format.Type = 22                 # 22kHz 16-bit mono
                fs.Open(tmp, 3, False)              # 3 = SSFMCreateForWrite
                old_out = _speaker.AudioOutputStream
                _speaker.AudioOutputStream = fs
                _speaker.Speak(text, 0)            # synchronous render to file
                _speaker.AudioOutputStream = old_out
                fs.Close()
                with open(tmp, "rb") as f:
                    data = f.read()
                return data if len(data) > 64 else None
            except Exception as e:
                print(f"[TTS] SAPI file capture error: {e}")
                return None
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def local_speak(text: str):
        """Speak on the server's own speaker (operator monitor)."""
        print(f"[PC] {text}")
        try:
            _speaker.Speak(text, 1)             # async local monitor
        except Exception:
            pass

    _sapi_available = True

except Exception:
    def _select_sapi_voice(substr: str):
        pass

    def sapi_to_wav(text: str) -> bytes | None:
        return None

    def local_speak(text: str):
        print(f"[PC] {text}")


# ── Piper setup (CLI; cross-platform) ─────────────────────────────────────────
_piper_cli = shutil.which("piper")


def piper_to_wav(text: str) -> bytes | None:
    """Render text → WAV bytes using the Piper CLI (neural, natural voice)."""
    if not (PIPER_MODEL and os.path.exists(PIPER_MODEL) and _piper_cli):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = tf.name
        subprocess.run(
            [_piper_cli, "--model", PIPER_MODEL, "--output_file", out_path],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        with open(out_path, "rb") as f:
            data = f.read()
        os.unlink(out_path)
        return data if data else None
    except Exception as e:
        print(f"[TTS] Piper error: {e}")
        return None


def synth_wav(text: str) -> tuple[bytes | None, str]:
    """Render TTS → (wav_bytes, engine_name) honoring TTS_ENGINE preference."""
    if TTS_ENGINE in ("auto", "piper"):
        wav = piper_to_wav(text)
        if wav:
            return wav, "piper"
        if TTS_ENGINE == "piper":
            return None, "piper_failed"
    wav = sapi_to_wav(text)
    return (wav, "sapi") if wav else (None, "none")


def init_tts() -> None:
    """Configure the SAPI voice and report which engines are available."""
    if _sapi_available:
        _select_sapi_voice(SAPI_VOICE)
        try:
            _speaker.Rate = SAPI_RATE
        except Exception:
            pass
    piper_ready = bool(PIPER_MODEL and os.path.exists(PIPER_MODEL) and _piper_cli)
    print(f"[TTS] Engine preference: {TTS_ENGINE}")
    print(f"[TTS] Piper ready: {piper_ready}"
          + ("" if piper_ready else f"  (cli={'yes' if _piper_cli else 'no'}, "
             f"model={'set' if PIPER_MODEL else 'unset'})"))
    print(f"[TTS] SAPI ready : {_sapi_available}")
    if not piper_ready and not _sapi_available:
        print("[TTS] WARNING: no PC voice available — client will use espeak.")