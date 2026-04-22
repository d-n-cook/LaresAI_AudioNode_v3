#pragma once

#include <Arduino.h>
#include "../lib/MCP23017/src/MCP23017.h"

// MCP23017 global instance (defined in main.cpp)
extern MCP23017 mcp;

// LED status control via MCP23017 — set to 1 when LED hardware is ready
#ifndef ENABLE_MCP_STATUS_LED
#define ENABLE_MCP_STATUS_LED 1FVVVVVV
#endif

#ifndef STATUS_LED_ACTIVE_LOW
#define STATUS_LED_ACTIVE_LOW 0
#endif

#ifndef STATUS_LED_MCP_RED_PIN
#define STATUS_LED_MCP_RED_PIN 10   // GPB2
#endif

#ifndef STATUS_LED_MCP_GREEN_PIN
#define STATUS_LED_MCP_GREEN_PIN 14  // GPB6
#endif

#ifndef STATUS_LED_MCP_BLUE_PIN
#define STATUS_LED_MCP_BLUE_PIN 15   // GPB7
#endif

// Per-channel output caps (0-7) for color balancing.
// Lowering G/B slightly often improves perceived color purity.
#ifndef STATUS_LED_RED_MAX
#define STATUS_LED_RED_MAX 7
#endif

#ifndef STATUS_LED_GREEN_MAX
#define STATUS_LED_GREEN_MAX 6
#endif

#ifndef STATUS_LED_BLUE_MAX
#define STATUS_LED_BLUE_MAX 6
#endif

extern void initStatusLedPwm();
extern void setLedColor(uint8_t r, uint8_t g, uint8_t b);  // 0–7 per channel

