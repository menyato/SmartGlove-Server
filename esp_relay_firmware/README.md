# ESP-01 (ESP8266) Relay Controller — Home Automation

Firmware for an **ESP-01 / ESP-01S** module driving a 2-channel relay board,
controlled remotely by the feature server as part of the Smart Glove
home-automation feature. The QR-based WiFi setup is scanned by the
**OrangePi's own camera** (reusing the same OpenCV capture code as OCR/Money),
not by the ESP — this board never needs a camera.

## Hardware

The ESP-01 only exposes two truly free GPIOs (GPIO0, GPIO2) — everything else
is VCC/GND/CH_PD/RST/TX(GPIO1)/RX(GPIO3). This firmware uses **GPIO2 and
GPIO1 (TX)** for the two relays:

| Relay module pin | ESP-01 pin |
|---|---|
| IN1 | GPIO2 |
| IN2 | GPIO1 (TX) |
| VCC | 5V — from a **separate supply**, not the ESP-01's own 3.3V regulator (share GND only) |
| GND | GND (shared with ESP-01) |

**Trade-off:** GPIO1 is the board's TX (serial console) pin. Once a relay is
wired there, this firmware never uses `Serial.print()` — there is no serial
debug output at all. Use the HTTP `/info` and `/status` endpoints instead
(see below) if you need to check what the board is doing.

Most cheap relay modules are **active-LOW** (a LOW signal energizes the
relay). If yours is active-HIGH, set `RELAY_ACTIVE_LOW false` in
`relay_controller.ino`. GPIO2 idles HIGH at boot (an ESP8266 boot-strapping
requirement) — with the default active-LOW wiring this lines up naturally
with "relay off," so there's no relay chatter at power-on.

## Flashing an ESP-01

The ESP-01 has no onboard USB or auto-reset circuit — you need a 3.3V
USB-to-serial adapter:

1. Wire adapter **TX → ESP-01 RX**, adapter **RX → ESP-01 TX**, plus 3.3V and GND.
2. Pull **GPIO0 to GND** before powering on / resetting the board — this puts
   it in flash mode. (GPIO2 must be left floating/HIGH, not grounded.)
3. Upload from the Arduino IDE, then disconnect GPIO0 from GND and
   power-cycle/reset to boot normally.

Arduino IDE settings:
- Board: **"Generic ESP8266 Module"**
- Flash Size: **"1MB (FS:none, OTA:~502KB)"** (no filesystem needed — this
  firmware only uses the small EEPROM-emulation flash sector)
- Reset Method: **"ck"** or **"nodemcu"**, whichever matches your adapter
- Upload Speed: **115200**

## Arduino IDE library setup

1. Install the **ESP8266 board package** (File → Preferences → Additional
   Boards Manager URLs → add
   `https://arduino.esp8266.com/stable/package_esp8266com_index.json`, then
   Boards Manager → search "esp8266" → install). This provides
   `ESP8266WiFi`, `ESP8266WebServer`, `ESP8266HTTPClient`, and `EEPROM` —
   nothing extra to install for those.
2. Install the **ArduinoJson** library (Library Manager → search
   "ArduinoJson", install v6.x).
3. Open `relay_controller/relay_controller.ino`, select board "Generic
   ESP8266 Module" with the settings above, and upload.

## First-time provisioning flow

The board ships unprovisioned — no config saved yet.

1. **On first boot** it starts its own WiFi access point:
   - SSID: `RelayCtrl-Setup`
   - Password: `relaysetup`
   - It runs a tiny HTTP server at `192.168.4.1`.

2. **On the server**, generate the setup QR code for this device (see
   `server_app/handlers/home.py` — `GET /qr/<device_id>` on the home-automation
   HTTP dashboard, default port 8090). This QR encodes:
   ```json
   {
     "type": "esp_relay_provision",
     "ssid": "<your home WiFi SSID>",
     "password": "<your home WiFi password>",
     "server_host": "<feature server LAN IP>",
     "server_home_port": 8090,
     "device_id": "relay01",
     "relay_count": 2
   }
   ```
   Display that QR on a phone or monitor screen. You can view it directly from
   any browser (including your phone) right now — no glove/camera needed just
   to check that it renders: `http://<server-lan-ip>:8090/qr/relay01`.

3. **On the glove**, trigger the Home Automation feature's "pair" voice
   command / gesture. It:
   - captures a camera frame and decodes the QR (`cv2.QRCodeDetector`),
   - temporarily switches the OrangePi's own WiFi to `RelayCtrl-Setup`,
   - `POST`s the decoded JSON to `http://192.168.4.1/provision`,
   - switches the OrangePi's WiFi back to the home network,
   - speaks a confirmation.

4. The ESP-01 saves `device_id`/`server_host`/`server_home_port` to its
   EEPROM-emulated flash, hands the WiFi SSID/password to `WiFi.begin()`
   (the ESP8266 SDK persists those itself — no extra code needed for that
   part), and reboots. It joins the home WiFi, starts its relay-control HTTP
   server, and registers itself with the feature server (`POST
   /esp/register`) so the server always knows its current IP — this repeats
   every 30 seconds as a heartbeat in case DHCP hands out a new address later.

If the saved WiFi credentials ever stop working (password changed, router
replaced), the board automatically falls back to the `RelayCtrl-Setup`
provisioning AP on the next boot instead of retrying forever — just repeat
step 2-3 with a fresh QR.

## HTTP API (once provisioned and connected)

| Method | Path | Description |
|---|---|---|
| GET | `/info` | `{"device_id","relay_count","fw","provisioned":true,"ip"}` |
| GET | `/relay?ch=1&state=on\|off\|toggle` | Switch one relay channel (ch 1 = GPIO2, ch 2 = GPIO1) |
| GET | `/status` | `{"relay1":"on\|off","relay2":"on\|off"}` |

The feature server is the only caller of these endpoints — the OrangePi never
talks to the ESP directly, per the chosen architecture (glove → feature
server → ESP over the home LAN).

### Quick manual test over WiFi (no glove needed)

Once the board shows up "online" at `http://<server-lan-ip>:8090/`, you can
toggle relays directly from a browser or curl to confirm the wiring works,
bypassing the glove entirely:
```
http://<esp-ip>/relay?ch=1&state=on
http://<esp-ip>/relay?ch=2&state=on
http://<esp-ip>/status
```
Or through the server (matches exactly what the glove sends):
```
curl "http://<server-lan-ip>:8090/" # confirm device_id + ip are listed first
```
then trigger a real toggle via `hub.py --relay1` / `--relay2` on the OrangePi,
or by POSTing the same message shape the glove uses to feature_server.py on
port 9000 (`{"feature":"home","action":"toggle","relay":1,"state":"on"}`).

## Re-pairing / resetting

There is no separate "factory reset" button wired up. To force
re-provisioning, either flash the board again (erasing flash, which also
clears the WiFi-credential and EEPROM areas) or add a physical reset button
pulling a spare pin low that clears `g_cfg.provisioned` and calls
`ESP.restart()` — left as a hardware extension since the current 2-relay
prototype PCB doesn't expose a spare pin for one (all of GPIO0/1/2/3 are
already committed to boot-mode-select, the two relays, and RX).
