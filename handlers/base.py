"""
handlers/base.py — server-side feature contract.

One handler instance is created PER CONNECTION PER FEATURE, so each handler can
hold its own session state. handle() takes the decoded message and returns a
response dict with at least {"tts": str, "quit": bool}; the dispatcher attaches
the synthesized audio.
"""


class FeatureHandler:
    name: str = "base"

    def handle(self, msg: dict) -> dict:
        raise NotImplementedError