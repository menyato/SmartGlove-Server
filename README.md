# Smart Glove — gesture hub + feature server

A gesture layer on the OrangePi that taps the ATmega sensor stream, recognizes
3 control gestures, and runs a scrollable feature menu. The first (and current)
feature is money recognition. The server is a dispatcher that routes each
feature's messages to a dedicated helper.

Your original `glove_controller.py`, `orangepi_client.py`, and `server.py` are
**reused unchanged** — imported as libraries. Nothing about the working STT,
camera, TTS, or YOLO logic was modified.

## Structure

```
smart_glove/
├── orangepi/                         # runs on the OrangePi 2W Zero
│   ├── hub.py                        # ← THE FIRST SCRIPT (entry point)
│   ├── glove_controller.py           # your driver (unchanged, used as a library)
│   ├── orangepi_client.py            # your money client (unchanged, used as a library)
│   ├── requirements_orangepi.txt
│   ├── gesture_hub/
│   │   ├── specs.py                  # GestureSpec + the 3 default gestures
│   │   ├── store.py                  # gestures.json load/save (defaults on failure)
│   │   ├── engine.py                 # SensorFrame stream -> gesture events
│   │   ├── recorder.py               # record-by-example (perform twice to save)
│   │   ├── registry.py               # the scrollable feature list
│   │   ├── feedback.py               # motor haptics + espeak prompts
│   │   └── state_machine.py          # Idle -> Menu -> Feature / Edit
│   ├── net/
│   │   └── client.py                 # ServerLink: one shared socket, per-feature msgs
│   └── features/
│       ├── base.py                   # Feature contract + FeatureContext
│       └── money_recognition.py      # wraps orangepi_client.py as a feature
└── server_app/                       # runs on the laptop
    ├── feature_server.py             # ← THE GENERAL SERVER (connect + route by feature)
    ├── tts.py                        # shared Piper/SAPI voice for ALL features
    ├── protocol.py                   # framed JSON recv/send
    └── handlers/
        ├── base.py                   # FeatureHandler contract
        └── money.py                  # self-contained money feature (YOLO + session + helpers)

The old monolithic server.py is no longer needed — feature_server.py + tts.py +
handlers/money.py replace it completely. The money detection/state-machine logic
is identical to the original, just moved into its own feature file.
```

## The 3 control gestures (re-recordable in-glove)

Finger indices: `0 Thumb · 1 Index · 2 Middle · 3 Ring · 4 Pinky`

| Gesture | Pose | Motion | Role |
|---------|------|--------|------|
| `START` | Thumb + Pinky | hold `tilt_right` | wake from idle · select / launch · exit a running feature |
| `NEXT`  | Thumb + Ring  | flick up (`tilt_backward`) | scroll the menu / scroll edit targets |
| `EDIT`  | Thumb + Index | flick down (`tilt_forward`) | enter edit mode · cancel edit |

**Flick = "from base to forward."** A flick fires on the rising edge of an
orientation flag — the flag was clear on the previous frame (the resting
baseline) and is now set. No firmware change and no raw-accel math; it uses the
`imu_flags` your firmware already sends. Re-baselining happens naturally because
each frame's previous state is the baseline for the next.

## Flow

```
Idle ──START──► Menu ──NEXT──► (scroll features)
                 │
                 ├──START (select)──► Feature runs (e.g. Money recognition)
                 │                      └─ voice "quit"/"done"  OR  START gesture ──► back to Menu
                 │
                 └──EDIT──► Edit: NEXT scrolls the gesture to change,
                                  START records it (perform twice to save),
                                  EDIT cancels.
```

A buzz confirms every recognized gesture (important for a blind user); the menu
and prompts are spoken via espeak. Inside money recognition you get the original
Piper/SAPI natural voice from the server.

## Per-feature messaging

The hub keeps **one** socket open. Every message carries a `feature` field:

```json
{ "feature": "money", "type": "voice", "text": "scan", "frame": "<base64 jpeg>" }
```

`feature_server.py` reads `feature`, looks it up in `FEATURE_HANDLERS`, and
forwards the message to that handler (which owns its own per-connection session
and runs the feature's helpers). A feature "quit" ends just that feature's
session — the socket stays open so the hub can launch another feature next.

## Adding a feature later

1. OrangePi: subclass `features.base.Feature` (set `name`/`title`, implement
   `run(ctx)`), and append an instance to `FEATURES` in `hub.py`.
2. Server: write `handlers/<name>.py` with a `FeatureHandler` subclass
   implementing `handle(msg)`, and register the class in `FEATURE_HANDLERS`
   in `feature_server.py`.
3. Both sides key off the same `name` — that's the whole contract. The general
   server voices every feature's reply through the shared `tts` module.

## Running

Server (laptop), from `server_app/`:

```
python feature_server.py --host 0.0.0.0 --port 9000 ^
    --tts auto --piper-model C:\path\to\voice.onnx --sapi-voice Zira ^
    --model C:\path\to\best.pt
```

The YOLO model loads lazily on the first scan, so the server starts instantly.

OrangePi:

```
cd orangepi
pip3 install --break-system-packages -r requirements_orangepi.txt
python3 hub.py --host <SERVER_IP> --port 9000 --uart /dev/ttyS5 --alsa plughw:0,0
```

Then: do `START` to enter the menu, `START` again to open money recognition,
and use it exactly as before (voice "scan", "redo", "discard", "quit"). Saying
"quit"/"done", or doing the `START` gesture, returns you to the menu.
```
```
