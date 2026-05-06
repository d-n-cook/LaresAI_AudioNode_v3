#pragma once
#include <Arduino.h>

// Checks GCS version manifest and applies OTA update if a newer version is available.
// Blocks until download completes — device reboots automatically on success.
// deviceType: "main" or "audionode"
// currentVersion: value of FIRMWARE_VERSION from version.h
// Returns false if no update needed or on error (caller can continue normally).
bool checkForOTAUpdate(const String& deviceType, const String& currentVersion);
