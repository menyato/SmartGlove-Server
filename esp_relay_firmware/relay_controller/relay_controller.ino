/*
 * relay_controller.ino — ESP-01 (ESP8266) home-automation relay board firmware.
 *
 * Board: ESP-01 / ESP-01S (ESP8266). Only 2 GPIOs are free on this module
 * (GPIO0, GPIO2) — everything else is VCC/GND/CH_PD/RST/TX(GPIO1)/RX(GPIO3).
 * This firmware uses GPIO2 and GPIO1 (the TX pin) for the two relays, per
 * your wiring — GPIO1 is safe to repurpose as a plain output (unlike GPIO0/
 * GPIO2, it has no boot-mode strapping role), but it means this board's
 * Serial console (which lives on GPIO1/TX) is NOT available once a relay is
 * wired there. There is no Serial.print() debugging in this firmware for
 * that reason — use the HTTP /info and /status endpoints instead.
 *
 * Wiring (2-channel relay module):
 *   Relay 1 IN  -> GPIO 2
 *   Relay 2 IN  -> GPIO 1 (TX — Serial debug output is unavailable while wired)
 *   Relay VCC   -> 5V (or 3.3V if your module supports it — NOT the ESP-01's
 *                  3.3V regulator output if using a 5V relay module; power the
 *                  relay board from a separate supply and share GND only)
 *   Relay GND   -> GND (shared with ESP-01 GND)
 * RELAY_ACTIVE_LOW below controls polarity: true = LOW turns a relay ON
 * (most common relay-module type), false = HIGH turns it ON. If "on" and
 * "off" behave backwards on your board, flip this one line and re-flash —
 * that's the only change needed. GPIO2 idles HIGH for a brief moment at
 * boot (an ESP8266 boot-strapping requirement, not something firmware can
 * change); with RELAY_ACTIVE_LOW=true this lines up with "off" at power-on,
 * with RELAY_ACTIVE_LOW=false Relay 1 may briefly flicker on for a fraction
 * of a second at boot before setup() corrects it — a hardware quirk of using
 * GPIO2 specifically, not a bug, and not present on Relay 2 (GPIO1).
 *
 * Flashing an ESP-01: you need a USB-to-serial adapter (3.3V!) wired
 * TX->RX, RX->TX, and GPIO0 pulled to GND *before* power-on/reset to enter
 * flash mode (release/reconnect GPIO0 and reset again to run normally
 * afterward). Arduino IDE board: "Generic ESP8266 Module", Flash Size
 * "1MB (FS:none OTA:~502KB)", Reset Method usually "ck" or "nodemcu"
 * depending on your adapter, Upload Speed 115200.
 *
 * Two states:
 *
 *   1) UNPROVISIONED (no config saved yet)
 *      Starts a SoftAP named AP_SSID (password AP_PASSWORD) and a tiny HTTP
 *      server on 192.168.4.1:
 *        GET  /info       -> {"device_id","relay_count","fw","provisioned":false}
 *        POST /provision  -> body JSON (see PROVISION JSON SCHEMA below).
 *                             WiFi credentials are handed to WiFi.begin()
 *                             (the ESP8266 SDK persists them to flash on its
 *                             own — no extra code needed for that part);
 *                             device_id/server_host/server_home_port are
 *                             saved to EEPROM-emulated flash here, then the
 *                             board reboots into station mode.
 *
 *   2) PROVISIONED (joins the home WiFi, controlled by the feature server)
 *      GET /info                       -> device_id, relay_count, fw, ip, provisioned:true
 *      GET /relay?ch=1&state=on        -> ch in {1,2}, state in {on,off,toggle}
 *      GET /status                     -> {"relay1":"on|off","relay2":"on|off"}
 *      Also calls home to the feature server's /esp/register endpoint right
 *      after connecting, and again every REGISTER_INTERVAL_MS as a heartbeat,
 *      so the server always has a fresh IP for this device (DHCP can change it).
 *
 * PROVISION JSON SCHEMA (posted to /provision — this exact shape is what the
 * server encodes into the setup QR code, and what features/home_automation.py
 * decodes from the QR and relays here — unchanged from the ESP32 version, so
 * nothing on the OrangePi or server side needs to change for this board):
 *   {
 *     "type":             "esp_relay_provision",
 *     "ssid":              "<home wifi ssid>",
 *     "password":          "<home wifi password>",
 *     "server_host":       "<feature server LAN IP>",
 *     "server_home_port":  8090,
 *     "device_id":         "relay01",
 *     "relay_count":       2
 *   }
 *
 * Required Arduino libraries (Library Manager):
 *   - ArduinoJson (by Benoit Blanchon), v6.x
 *   - ESP8266 board package (adds ESP8266WiFi, ESP8266WebServer,
 *     ESP8266HTTPClient, EEPROM — all ship with it, nothing extra to install)
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <EEPROM.h>
#include <ArduinoJson.h>

// ── Hardware config ───────────────────────────────────────────────────────────
#define RELAY1_PIN 2   // GPIO2
#define RELAY2_PIN 1   // GPIO1 / TX — Serial is unavailable once this is wired
#define RELAY_ACTIVE_LOW false  // false = HIGH turns the relay ON
#define RELAY_COUNT 2
#define FW_VERSION "1.0.0-esp01"

// ── Provisioning AP config ────────────────────────────────────────────────────
#define AP_SSID     "RelayCtrl-Setup"
#define AP_PASSWORD "relaysetup"        // >=8 chars required by WiFi.softAP
#define WIFI_CONNECT_TIMEOUT_MS 20000
#define REGISTER_INTERVAL_MS    30000

// ── EEPROM-backed config (WiFi creds themselves are persisted separately by
//    the ESP8266 SDK via WiFi.begin() — WiFi.persistent(true) is the default) ──
#define CONFIG_MAGIC 0x52454C31UL   // "REL1", sanity check for a blank/foreign flash
struct Config {
  uint32_t magic;
  bool     provisioned;
  char     device_id[32];
  char     server_host[64];
  uint16_t server_home_port;
};
Config g_cfg;

ESP8266WebServer server(80);

uint32_t g_lastRegisterMs = 0;
bool relayState[RELAY_COUNT + 1] = {false, false, false}; // index 1..2 used

// ── Relay control ──────────────────────────────────────────────────────────────
int relayPin(int ch) { return ch == 1 ? RELAY1_PIN : RELAY2_PIN; }

void applyRelay(int ch, bool on) {
  relayState[ch] = on;
  bool level = RELAY_ACTIVE_LOW ? !on : on;
  digitalWrite(relayPin(ch), level ? HIGH : LOW);
}

// ── EEPROM load/save ───────────────────────────────────────────────────────────
void loadConfig() {
  EEPROM.begin(sizeof(Config));
  EEPROM.get(0, g_cfg);
  EEPROM.end();
  if (g_cfg.magic != CONFIG_MAGIC) {
    // Blank/foreign flash — reset to safe defaults
    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.magic = CONFIG_MAGIC;
    g_cfg.provisioned = false;
    strncpy(g_cfg.device_id, "relay01", sizeof(g_cfg.device_id) - 1);
    g_cfg.server_home_port = 8090;
  }
}

void saveConfig() {
  EEPROM.begin(sizeof(Config));
  EEPROM.put(0, g_cfg);
  EEPROM.commit();
  EEPROM.end();
}

// ── Common JSON helpers ────────────────────────────────────────────────────────
void sendJson(int code, JsonDocument &doc) {
  String body;
  serializeJson(doc, body);
  server.send(code, "application/json", body);
}

// ── Provisioning-mode routes ───────────────────────────────────────────────────
void handleInfoUnprovisioned() {
  JsonDocument doc;
  doc["device_id"]   = g_cfg.device_id;
  doc["relay_count"] = RELAY_COUNT;
  doc["fw"]          = FW_VERSION;
  doc["provisioned"] = false;
  sendJson(200, doc);
}

// Shared by both provisioning routes below. Saves config, hands WiFi creds
// to the SDK, acks the request, then reboots into station mode.
void doProvision(const String &ssid, const String &pass, const String &host,
                 uint16_t port, const String &devId) {
  g_cfg.provisioned = true;
  strncpy(g_cfg.device_id, devId.c_str(), sizeof(g_cfg.device_id) - 1);
  strncpy(g_cfg.server_host, host.c_str(), sizeof(g_cfg.server_host) - 1);
  g_cfg.server_home_port = port;
  saveConfig();

  // WiFi.begin() with persistent mode (default) makes the SDK itself save
  // ssid/pass to flash — no separate storage needed for those two fields.
  WiFi.persistent(true);
  WiFi.begin(ssid.c_str(), pass.c_str());

  server.send(200, "application/json", "{\"ok\":true}");
  delay(500);
  ESP.restart();
}

// POST /provision — JSON body, used by the glove's automated pairing flow
// (features/home_automation.py decodes the QR then POSTs it here directly).
void handleProvision() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"no body\"}");
    return;
  }
  JsonDocument in;
  DeserializationError err = deserializeJson(in, server.arg("plain"));
  if (err) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"bad json\"}");
    return;
  }
  String ssid   = in["ssid"]        | "";
  String pass   = in["password"]    | "";
  String host   = in["server_host"] | "";
  uint16_t port = in["server_home_port"] | 8090;
  String devId  = in["device_id"]   | String(g_cfg.device_id);

  if (ssid.length() == 0 || host.length() == 0) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"missing ssid/host\"}");
    return;
  }
  doProvision(ssid, pass, host, port, devId);
}

// GET /provision?ssid=...&password=...&server_host=...&server_home_port=...&device_id=...
// Manual-testing convenience only (e.g. from a phone browser after joining
// RelayCtrl-Setup by hand, with no HTTP-client app) — same effect as the
// POST route above, just reachable by tapping a plain URL.
void handleProvisionGet() {
  if (!server.hasArg("ssid") || !server.hasArg("server_host")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"missing ssid/server_host\"}");
    return;
  }
  String ssid   = server.arg("ssid");
  String pass   = server.hasArg("password") ? server.arg("password") : "";
  String host   = server.arg("server_host");
  uint16_t port = server.hasArg("server_home_port") ? server.arg("server_home_port").toInt() : 8090;
  String devId  = server.hasArg("device_id") ? server.arg("device_id") : String(g_cfg.device_id);
  doProvision(ssid, pass, host, port, devId);
}

// ── Connected-mode routes ──────────────────────────────────────────────────────
void handleInfoConnected() {
  JsonDocument doc;
  doc["device_id"]   = g_cfg.device_id;
  doc["relay_count"] = RELAY_COUNT;
  doc["fw"]          = FW_VERSION;
  doc["provisioned"] = true;
  doc["ip"]          = WiFi.localIP().toString();
  sendJson(200, doc);
}

void handleRelay() {
  if (!server.hasArg("ch")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"missing ch\"}");
    return;
  }
  int ch = server.arg("ch").toInt();
  if (ch < 1 || ch > RELAY_COUNT) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"bad channel\"}");
    return;
  }
  String state = server.hasArg("state") ? server.arg("state") : "toggle";
  bool newState;
  if (state == "on")           newState = true;
  else if (state == "off")     newState = false;
  else /* toggle */            newState = !relayState[ch];

  applyRelay(ch, newState);

  JsonDocument doc;
  doc["ok"]    = true;
  doc["relay"] = ch;
  doc["state"] = newState ? "on" : "off";
  sendJson(200, doc);
}

void handleStatus() {
  JsonDocument doc;
  for (int ch = 1; ch <= RELAY_COUNT; ch++) {
    doc[String("relay") + ch] = relayState[ch] ? "on" : "off";
  }
  sendJson(200, doc);
}

// ── Server registration (heartbeat) ───────────────────────────────────────────
void registerWithServer() {
  if (strlen(g_cfg.server_host) == 0) return;
  WiFiClient client;
  HTTPClient http;
  String url = "http://" + String(g_cfg.server_host) + ":" + String(g_cfg.server_home_port) + "/esp/register";
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");

  JsonDocument doc;
  doc["device_id"]   = g_cfg.device_id;
  doc["ip"]          = WiFi.localIP().toString();
  doc["relay_count"] = RELAY_COUNT;
  doc["fw"]          = FW_VERSION;
  String body;
  serializeJson(doc, body);

  http.POST(body);
  http.end();
  g_lastRegisterMs = millis();
}

// ── Setup / loop ───────────────────────────────────────────────────────────────
void startProvisioningAP() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);

  server.on("/info", HTTP_GET, handleInfoUnprovisioned);
  server.on("/provision", HTTP_POST, handleProvision);
  server.on("/provision", HTTP_GET, handleProvisionGet);
  server.begin();
}

bool connectStation() {
  WiFi.mode(WIFI_STA);
  WiFi.begin();   // no args — reconnects using SDK-persisted credentials

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
  }
  return WiFi.status() == WL_CONNECTED;
}

void startConnectedServer() {
  server.on("/info", HTTP_GET, handleInfoConnected);
  server.on("/relay", HTTP_GET, handleRelay);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();
}

void setup() {
  pinMode(RELAY1_PIN, OUTPUT);
  pinMode(RELAY2_PIN, OUTPUT);
  applyRelay(1, false);
  applyRelay(2, false);

  loadConfig();

  if (!g_cfg.provisioned) {
    startProvisioningAP();
    return;
  }

  if (connectStation()) {
    startConnectedServer();
    registerWithServer();
  } else {
    // Saved credentials no longer work (wrong password, network moved, etc.)
    // — fall back to provisioning mode instead of retrying forever.
    startProvisioningAP();
  }
}

void loop() {
  server.handleClient();

  // Only relevant once connected — re-announce IP periodically in case DHCP
  // handed out a new one, and so the server's device list doesn't go stale.
  if (g_cfg.provisioned && WiFi.status() == WL_CONNECTED &&
      millis() - g_lastRegisterMs > REGISTER_INTERVAL_MS) {
    registerWithServer();
  }
}
