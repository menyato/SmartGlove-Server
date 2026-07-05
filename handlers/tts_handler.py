"""
handlers/tts_handler.py — standalone TTS feature.

Hub sends {"feature": "tts", "text": "..."}.
Server returns {"tts": text, ...} → _send_with_audio() synthesises a WAV
via SAPI/Piper and attaches it as "audio" (base64).
Hub reads resp["audio"] and plays it with fb.play_raw().
"""

from handlers.base import FeatureHandler


class TTSHandler(FeatureHandler):
    name = "tts"

    def handle(self, msg: dict) -> dict:
        text = msg.get("text", "").strip()
        if not text:
            return {"tts": "", "ok": True, "quit": False}
        return {"tts": text, "ok": True, "quit": False}
