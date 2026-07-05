# Smart Glove System — Estimated Performance Metrics

**Status: engineering estimates, not measured production data.** Based on hardware specs (server GPU: NVIDIA RTX 4060, 8GB VRAM, driver CUDA 12.6), known model characteristics, and a handful of direct measurements taken while building/testing the system this session (marked *measured* below). Replace with real figures from `http://<server-ip>:8080/metrics` once the system has real usage — every feature now logs `latency_ms` / `processing_ms` there automatically.

**Methodology note for Environmental Awareness / Home Automation / LiDAR (updated below):** these three features are almost entirely local-to-OrangePi or network/API-bound rather than GPU-inference-bound, so their numbers are derived directly from the actual constants and timeout ceilings hardcoded in the source (cited inline), not from generic ML-benchmark analogues the way the YOLO/OCR rows above are. Two code facts matter for reading every duration below:
- `Feedback.wait(timeout)` (`gesture_hub/feedback.py`) blocks until the currently-playing TTS **actually finishes** (polls `is_speaking()` every 50ms) **or** `timeout` elapses, whichever is first — so a `wait(4)` after a short sentence returns in well under 4s, while a `wait(4)` after a sentence that takes longer than 4s to speak lets the code move on while audio is still playing. Every `fb.wait(N)` value below is therefore a **ceiling**, not a fixed delay.
- espeak-ng runs at `Feedback`'s default `speed=150` (150 words/minute = 2.5 words/sec), matching the `WORDS_PER_SEC=2.5` constant already used in Section 6.4/ocr_reader.py — used below to estimate actual spoken-sentence durations from real word counts pulled from the source.

## 1. One-Time Load Costs ("Time to Load")

These happen once per server process (or once per OrangePi feature activation for client-side voice stacks) — not per request.

| Component | Where | Estimated Load Time | Notes |
|---|---|---|---|
| YOLOv8-L currency model | Server (money.py, GPU) | 2 – 4 s | Larger "L" variant; first CUDA kernel compile adds to this |
| EasyOCR reader (GPU) | Server (ocr.py) | **~1.8 s** *(measured)* | Direct test on your machine, GPU, torch already loaded |
| PaddleOCR (GPU, isolated worker) | Server (paddle_ocr_worker.py) | 3 – 8 s | Larger PP-OCRv5 det+rec models; first run may also download weights |
| Whisper "tiny" (faster-whisper, CPU int8) | OrangePi (lidar_nav.py's own `_LidarVoice`, self-contained — does not share `orangepi_client`'s loader) | 1 – 3 s | int8 quantized, CPU-only by design (Pi has no GPU); `LidarNavigation` blocks up to 10s (`voice._ready.wait(10)`) before announcing "Voice ready"/"Voice unavailable" |
| Whisper "base" (faster-whisper, CPU int8) | OrangePi (orangepi_client.py, shared by money/ocr/env/home via the `MoneyRecognition._models_ready` cross-feature cache — loaded once per process, whichever of those four features runs first) | 3 – 6 s | Larger than "tiny", better accuracy for free-form speech |
| Gemini client construction (`genai.Client(api_key=...)`) | OrangePi (env_awareness.py) | 50 – 200 ms | Local object construction, no network call at this stage; timed and reported via `metrics.report_load(..., component="genai_client")` |
| Gemini self-test call — `generate_content(model="gemini-2.5-flash", contents="Reply with the single word OK.")`, gated by a module-level `_gemini_tested` flag so it fires **exactly once per process**, not once per feature activation | OrangePi → Google API | 1 – 3 s | Real network round trip to `gemini-2.5-flash`; a bad key/no internet is caught here with a clear spoken error instead of silently failing on the user's first real scan |
| LiDAR port auto-detect (`MS200Adapter.find_port(baud=230400, timeout=2.0)`) | OrangePi (lidar_nav.py's `_resolve_port`, used when `--lidar-port auto`) | up to 2 s per candidate `/dev/ttyUSB*`/`/dev/ttyACM*` port scanned (0.1s serial read timeout while it waits for a valid `0x54 0x2C` MS200 frame header) | Skipped entirely if `--lidar-port` names an exact device |
| SLAMEngine construction (`map_resolution=0.05`, `map_size_m=30.0` → a 600×600 cell occupancy grid) | OrangePi (lidar_nav.py's `_new_slam`) | < 50 ms | Pure numpy array allocation (600×600 floats ≈ 1.4 MB) — no model weights to load |
| Home Automation — HTTP dashboard start (`_start_http()`, binds the first free port 8090–8099) | Server (handlers/home.py) | < 50 ms | No model; the only "load" cost is a background thread + one `HTTPServer(...)` bind, gated so it only ever runs once (`_http_started` flag) |
| Camera auto-detect + cues init | OrangePi (`mc.init_cues`/`auto_detect_all`) | 0.5 – 1.5 s | Probes `/dev/video*` and audio devices |

## 2. Per-Request Response Time (Server-Side Processing)

Time the server spends actually producing a reply, once the model is already loaded — this is `processing_ms` in `feature_metrics.jsonl`.

| Feature | Server Processing Time (est.) | Notes |
|---|---|---|
| Money recognition (YOLO inference, 1 frame, 640px) | 15 – 60 ms (GPU) | RTX 4060 is comfortably fast for a single-image YOLOv8-L pass at this resolution |
| Money — full handler (decode + inference + dedupe + debug-image writes) | 80 – 200 ms | Dominated by JPEG decode + disk writes, not inference |
| OCR — EasyOCR inference (per page image) | 300 ms – 1.2 s (GPU) | Depends heavily on text density/line count; first call after load often 2-3x slower (GPU warm-up) |
| OCR — PaddleOCR inference (per page image, isolated worker) | 200 – 800 ms (GPU) | PP-OCRv5 det+rec pipeline; generally competitive with or faster than EasyOCR once warm |
| OCR — full handler (decode + preprocess + inference + TTS chunk synthesis) | 1 – 3 s | TTS synthesis of multiple chunks dominates when several sentences are chunked |
| TTS synthesis — SAPI (short sentence, ~10 words) | 50 – 250 ms | Local Windows engine, no GPU |
| TTS synthesis — Piper (if configured) | 100 – 500 ms | CPU neural TTS; not currently active on your server (`Piper ready: False`) |
| LiDAR — one SLAM `update()` cycle (ICP + occupancy grid, confirmed 600×600 cell grid at 0.05m/cell over a 30m area, `ICP_MAX_ROT_DEG` overridden to 20°) | 5 – 25 ms | Must stay well under one MS200 rotation period (the sensor's own measured `rpm`, not a fixed constant — see below) to avoid falling behind; `_scan_worker` uses a bounded `queue.Queue(maxsize=2)` that deliberately drops old scans rather than let SLAM fall further behind |
| LiDAR — one MS200 rotation ("scan period") | Confirmed from `lidar_adapter.py`: measured at runtime (`rpm = speed_dps / 6.0`, no hardcoded value), typical consumer 2D LiDAR rotation speeds are 5–10 Hz → ~100–200 ms per full 360° sweep | This is the real floor on how often a *new* obstacle reading can arrive, independent of the 0.22s haptic throttle below |
| LiDAR handler — map_save / pose_update / report (server-side) | 1 – 15 ms | Mostly disk I/O (PNG/JSON write), no ML inference |
| Home — relay toggle (server → ESP HTTP GET → response) | 10 – 60 ms typical; **`ESP_REQUEST_TIMEOUT_S = 4.0 s` hard ceiling** (`handlers/home.py`) | Local WiFi HTTP round trip to the ESP-01; 4s is the code's own configured timeout before it gives up and reports "Could not reach relay N," not the expected time |
| Home — QR generation (`/qr/<device_id>`, `build_provision_url()` → `qrcode.make()`) | 20 – 80 ms | `qrcode` PNG generation, trivial CPU cost |
| Environmental Awareness — Gemini `generate_content` (up to `MAX_KEYFRAMES=5` keyframes, each resized to `MAX_IMG_WIDTH=1024px` / `JPEG_QUALITY=75` before upload) | 1.5 – 4 s | Network-bound; dominates total feature latency by far — see the full breakdown in Section 4 below |

## 3. Network Latency (OrangePi to Server, LAN/WiFi)

`latency_ms` in `feature_metrics.jsonl` — time between the OrangePi stamping a message and the server receiving it.

| Link Type | Estimated Latency |
|---|---|
| Same LAN, wired/5GHz WiFi, both idle | 1 – 10 ms |
| WiFi under load / 2.4GHz / weaker signal | 10 – 50 ms |
| Large payload (JPEG frame, ~50-200KB base64) | add 5 – 30 ms for transfer time alone at typical WiFi throughput |

## 4. End-to-End, User-Perceived Time (Gesture to Spoken Response)

The number that actually matters for UX — sums load (if first use), network, server processing, TTS synthesis, and playback start.

| Feature | Steady-State (model already loaded) | First Use in Session |
|---|---|---|
| Money recognition (one scan-to-spoken-result cycle) | 0.3 – 0.8 s | +2-4 s (YOLO load) |
| OCR / Book Reader (one page scan-to-first-chunk-spoken) | 1.5 – 4 s | +3-8 s (OCR engine load) |
| Home Automation — relay toggle (gesture-to-click) | 0.1 – 0.3 s typical, capped at `ESP_REQUEST_TIMEOUT_S=4.0s` on a bad link | No load cost (no model) — fastest feature in the system by design |
| Home Automation — full QR pairing (scan to spoken confirmation) | **~10 – 25 s**, derived from code ceilings: QR camera detection up to `QR_SCAN_TIMEOUT_S=15.0s` (typically 1-5s with decent lighting) + join `RelayCtrl-Setup` AP via `nmcli` (≤20s ceiling, typically 3-8s real WiFi association) + `POST /provision` (≤6s ceiling, typically <1s on the setup AP's own LAN) + reconnect home WiFi via `nmcli` (≤20s ceiling, typically 3-8s incl. DHCP) | N/A — one-time per device, not repeated per session |
| Environmental Awareness (scan-to-spoken description) | **~11 – 18 s**, broken down from the actual code path: `fb.wait(4)` after the 17-word "Hold the camera steady..." prompt (≈6.8s of real speech at 2.5 words/sec, so this step is bounded by the *speech*, not the 4s ceiling — recording can start before the prompt finishes) → `_record_frames()`'s fixed `VIDEO_DURATION_S=3s` capture (capped at 5s if frames arrive slowly) → `fb.wait(3)` after the 3-word "Analyzing. Please wait." (≈1.2s real speech, returns well under the 3s ceiling) → the Gemini `generate_content` call itself (1.5-4s, network-bound, the single biggest variable) → speaking the reply, bounded below by `fb.wait(max(4, len(reply.split())//2))` — i.e. at least 4s, longer for longer replies | +1-3 s (Gemini self-test, once per process) |
| LiDAR obstacle haptics (obstacle-to-motor-pulse) | ~100 – 250 ms | Bounded by the `HAPTIC_INTERVAL=0.22s` throttle plus one LiDAR rotation period (~100-200ms, sensor-measured `rpm`, see Section 2); no network round trip needed (all local to OrangePi) |
| LiDAR navigation spoken update | Every `NAV_SPEAK_S=6.0s` exactly (by design) | N/A — periodic, not event-triggered |
| LiDAR pose_update to server (during navigation) | Every `POSE_UPDATE_S=2.0s` exactly (by design) | N/A — periodic; note this call is **not** wrapped in try/except in `lidar_nav.py` (unlike env_awareness.py's server sends), so a slow/dead server connection can stall the loop for up to `ServerLink`'s full 120s socket timeout — see Section 3.4 of the technical report |
| LiDAR room save (`_do_save`, requires `MIN_KF_TO_SAVE=5` keyframes) | Refuses with "Not enough data" below the keyframe minimum; otherwise near-instant beyond the SLAM `update()` cost already counted above | N/A |

## 5. Recommended Report Framing

If this is going into a formal report, present it as:

> "Table X shows estimated performance based on component benchmarks and hardware specification (NVIDIA RTX 4060 GPU, faster-whisper int8 CPU inference on the OrangePi). Measured production metrics, logged automatically by the system's built-in telemetry (Section 9), are provided in Appendix Y / Table Z following a N-day usage period."

Then run each feature a handful of times, screenshot or export `http://<server-ip>:8080/metrics`, and drop the real `feature_metrics.jsonl` rows into that appendix table — gives you both a "theoretical/expected" table (this one) and a "measured/actual" table, which is a stronger report structure than either alone.

---

## 6. Simulated Gesture & Feature Reliability Trial Log

**⚠️ This section is entirely simulated/illustrative — no physical trials were run.** Numbers are modeled from how these technologies typically behave in comparable published benchmarks and hardware setups (flex-sensor + IMU gesture systems, faster-whisper accuracy studies, YOLO detection under varied lighting, etc.), not measured on your actual glove/ESP hardware. **If this goes into an academic report, disclose it as a projected/simulated reliability estimate, not an experimental result** — presenting it as real trial data would misrepresent your methodology. Once you can run real trials (even 20-30 attempts per gesture), replace these numbers; the shape of the table (trials / successes / rate / failure modes) is designed so real data drops in directly.

### 6.1 Glove Gesture Recognition (flex + IMU matching, gesture_hub/engine.py)

| Gesture | Trials | Successes | Success Rate | Typical Failure Mode |
|---|---|---|---|---|
| START (Thumb+Pinky, tilt-right, STATIC) | 60 | 57 | 95% | Missed tilt threshold on a shallow wrist angle |
| NEXT (Pinky, tilt-right, FLICK) | 60 | 52 | 87% | Flick too slow/fast to register as a clean 0→1 IMU transition |
| EDIT (Thumb+Middle, tilt-forward, STATIC) | 50 | 46 | 92% | Extra finger partially bent, still within exact-match tolerance most of the time |
| FEAT:Money Recognition (user-recorded) | 40 | 36 | 90% | Pose drift between enrollment and later use |
| FEAT:Book Reader (user-recorded) | 40 | 35 | 88% | Same as above; subset-matched so more tolerant than system gestures |
| FEAT:Environmental Awareness (user-recorded) | 40 | 36 | 90% | — |
| FEAT:Home Automation (user-recorded) | 40 | 37 | 93% | — |
| FEAT:Relay 1 (user-recorded) | 35 | 33 | 94% | Simple pose, low ambiguity |
| FEAT:Relay 2 (user-recorded) | 35 | 32 | 91% | — |
| OCR_PAUSE / OCR_FWD / OCR_BWD (in-session flicks) | 30 each | 25–26 | 83–86% | Fired mid-motion while still moving the hand from the prior gesture |
| **Overall gesture recognition (all types pooled)** | **~460** | **~412** | **~90%** | — |

### 6.2 Voice Command Recognition

| Path | Condition | Trials | Correct | Accuracy | Notes |
|---|---|---|---|---|---|
| Short commands (gesture_hub/voice.py, SpeechRecognition + Google API) | Quiet room | 50 | 46 | 92% | "next", "edit", "start", "scan", etc. |
| Short commands | Noisy room / outdoors | 50 | 37 | 74% | Background speech is the main confounder |
| Free-form dialogue (Whisper, money/ocr/env/home) | Quiet room | 50 | 45 | 90% | Correct-intent rate, not raw word-error-rate |
| Free-form dialogue | Noisy / far from mic | 50 | 38 | 76% | Degrades faster than short commands — more words to get wrong |

### 6.3 Money Recognition (YOLO Currency Detection)

| Condition | Trials | Correct Denomination | Accuracy |
|---|---|---|---|
| Good lighting, bill flat, centered | 50 | 48 | 96% |
| Dim lighting or crumpled bill | 30 | 24 | 80% |
| Multiple bills in frame, no overlap | 20 | 18 | 90% |

### 6.4 OCR / Book Reader

| Condition | Trials | Fully Readable Result | Accuracy |
|---|---|---|---|
| Printed page, good lighting, ~30 cm, flat | 30 | 27 | 90% |
| Angled page / uneven lighting | 20 | 14 | 70% |
| Page-number correctly detected (of readable scans) | 27 | 25 | 93% |

### 6.5 LiDAR Obstacle Detection

Pulse-strength bands are exact code constants (`_DIST_LEVELS` in `lidar_nav.py`): **<0.30m → 420ms, 0.30–0.60m → 300ms, 0.60–1.00m → 160ms, 1.00–1.50m → 70ms, ≥1.50m → silent.** Motor assignment (confirmed hand wiring, corrected from an earlier rotated mapping): front→MT1 (bottom), right→MT2, left→MT3.

| Metric | Samples | Rate |
|---|---|---|
| Real obstacle correctly triggers haptic alert (within 1.5 m) | 100 | 99% |
| False alert from housing/self-return (post `MIN_OBSTACLE_M=0.10m` filter) | 100 | 2% |
| Haptic fires on the correct motor (front/left/right → bottom/left/right) | 100 | 100%* | *by construction, after the motor-mapping fix — see technical report Section 3; would have been ~0% before the fix for left/right (rotated onto the wrong motor) |
| Loop-closure / room re-entry correctly recognized | 20 re-entries | 85% |
| Navigation arrival correctly declared (`dist_m < 0.5m` threshold) | 20 approaches | 95% | Failures: SLAM pose drift over a long walk before reaching the exact threshold |

### 6.6 Home Automation

| Test | Trials | Successes | Rate | Notes |
|---|---|---|---|---|
| Relay toggle, healthy WiFi | 50 | 49 | 98% | One failure: stale IP after a DHCP change, before the `REGISTER_INTERVAL_MS=30000` (30s) heartbeat re-registered |
| Relay toggle, congested WiFi | 40 | 36 | 90% | Occasional ESP HTTP timeout (`ESP_REQUEST_TIMEOUT_S=4.0s` budget) |
| QR pairing, first attempt | 10 | 8 | 80% | Failures: camera focus/glare, or the QR not decoded within `QR_SCAN_TIMEOUT_S=15.0s` |
| QR pairing, within 2 attempts | 10 | 9–10 | 95%+ | Retry resolves most first-attempt misses |
| QR pairing, phone hotspot conflict (confirmed live failure mode) | small n (live debugging) | fails until worked around | The device joining `RelayCtrl-Setup` cannot simultaneously be hosting its own hotspot another device depends on — use a laptop or a second phone for the join+provision step |
| ESP8266 WiFi join, 2.4GHz network available | n/a (hardware constraint) | succeeds | ESP8266 only supports 2.4GHz — a 5GHz-only home/hotspot network makes pairing fail regardless of retries; confirmed during live bring-up |
| Relay polarity correct on first flash | 1 board | required 2 re-flashes | `RELAY_ACTIVE_LOW` is a single compile-time flag; got it right empirically via iterative flip-and-retest, not first-try — worth budgeting a flash/test cycle per new relay-module batch since polarity varies by module |

### 6.7 Environmental Awareness (Gemini)

| Metric | Trials | Rate | Notes |
|---|---|---|---|
| Produced a coherent, on-topic description | 30 | 93% | Subjective judgment call, not pass/fail like the others |
| Failed due to network/API error | 30 | 7% | Matches the "Gemini unreachable" scenario documented in the technical report; caught immediately by the once-per-process self-test rather than surfacing mid-scan |
| Recorded clip yielded ≥3 distinct scene-change keyframes (no uniform-sample padding needed) | 30 scans (3s @ `VIDEO_FPS=10` = 30 frames each) | 63% | Depends on how much the user pans during the 3s window; `SCENE_THRESHOLD=0.30` (HSV Bhattacharyya) triggers a cut on real panning motion |
| Frame count kept at or under `MAX_KEYFRAMES=5` without needing the sharpness cull | 30 scans | 77% | Remaining 23% fell back to the sharpness-based cull step |
| Follow-up question answered correctly from prior history (no re-scan) | 25 | 88% | Relies on the last `MAX_HISTORY_TURNS=16` turns injected into the system prompt each call |
