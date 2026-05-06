#include "LaresOTA.h"
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <HTTPUpdate.h>
#include <ArduinoJson.h>

static const char* VERSIONS_URL =
    "https://storage.googleapis.com/lares-firmware/versions.json";

bool checkForOTAUpdate(const String& deviceType, const String& currentVersion) {
    if (WiFi.status() != WL_CONNECTED) return false;

    Serial.printf("[OTA] Checking (device=%s current=%s)\n",
                  deviceType.c_str(), currentVersion.c_str());

    // --- Fetch version manifest ---
    WiFiClientSecure client;
    client.setInsecure();

    HTTPClient http;
    if (!http.begin(client, VERSIONS_URL)) {
        Serial.println("[OTA] Failed to open version URL");
        return false;
    }
    http.setTimeout(10000);
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[OTA] Version check HTTP %d\n", code);
        http.end();
        return false;
    }

    StaticJsonDocument<512> doc;
    auto err = deserializeJson(doc, http.getStream());
    http.end();
    if (err || !doc.containsKey(deviceType)) {
        Serial.printf("[OTA] Manifest parse error or device type missing\n");
        return false;
    }

    String latestVersion = doc[deviceType]["version"] | "";
    String firmwareUrl   = doc[deviceType]["url"]     | "";

    if (latestVersion.isEmpty() || firmwareUrl.isEmpty()) {
        Serial.println("[OTA] Manifest missing version or url");
        return false;
    }
    if (latestVersion == currentVersion) {
        Serial.printf("[OTA] Up to date (%s)\n", currentVersion.c_str());
        return false;
    }

    Serial.printf("[OTA] Update available: %s -> %s\n",
                  currentVersion.c_str(), latestVersion.c_str());
    Serial.printf("[OTA] Downloading: %s\n", firmwareUrl.c_str());

    // --- Download and apply ---
    WiFiClientSecure updateClient;
    updateClient.setInsecure();

    httpUpdate.setLedPin(-1);
    httpUpdate.rebootOnUpdate(true);

    HTTPUpdateResult result = httpUpdate.update(updateClient, firmwareUrl);

    switch (result) {
        case HTTP_UPDATE_OK:
            // rebootOnUpdate=true means we never reach here
            Serial.println("[OTA] Applied — rebooting");
            return true;
        case HTTP_UPDATE_NO_UPDATES:
            Serial.println("[OTA] Server says no updates");
            return false;
        default:
            Serial.printf("[OTA] Failed (%d): %s\n",
                          httpUpdate.getLastError(),
                          httpUpdate.getLastErrorString().c_str());
            return false;
    }
}
