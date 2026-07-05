"""
handlers/metrics_handler.py — generic sink for client-side (OrangePi) timing
reports: model/voice-stack load time, capture time, anything a feature wants
to record that happens before/outside a normal request-response round trip
(that round-trip latency is already captured centrally in feature_server.py's
dispatch loop for every feature automatically — see _log_feature_metric).

OrangePi side: OrangePI/metrics.py's report_metric() sends these, fire-and-
forget, tagged {"feature": "metrics"}.

Message shape:
  {"event": "client_load"|"client_action", "source_feature": "<name>",
   "ms": <float>, ...any extra fields the caller wants logged}

Always replies silently ({"tts": "", "quit": False}) — this is telemetry,
never spoken.
"""

import json
import os
import threading
import time

from handlers.base import FeatureHandler

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, "client_metrics.jsonl")
_lock = threading.Lock()


def _log(rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with _lock:
        try:
            with open(METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


class MetricsHandler(FeatureHandler):
    name = "metrics"

    def handle(self, msg: dict) -> dict:
        rec = {k: v for k, v in msg.items() if k != "feature"}
        _log(rec)
        return {"tts": "", "quit": False}
