#include <Arduino.h>
#include "driver/i2s.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <time.h>
#include <WiFi.h>
#include <WebServer.h>
#include "WebSocketsServer.h"
#include "ElegantOTA.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "../lib/MCP23017/src/MCP23017.h"
#include "globals.h"

// PDM Microphone pin mapping for ESP32-S3-Sense board
#define I2S_SCK 42   // PDM clock pin for MIC
#define I2S_SD 41    // PDM data pin for MIC

#define SD_CS_PIN 21 // SD Card 

const int BUZZER  = 4; 

#define SAMPLE_RATE 16000
#define FRAME_SAMPLES 64
#define WIDTH 60

#define EI_INFERENCE_MIN_CONFIDENCE 0.60f
#define EI_RESULT_PRINT_INTERVAL_MS 2000

int16_t eiInferenceSlice[EI_CLASSIFIER_SLICE_SIZE];
size_t eiSliceWriteIndex = 0;
bool eiClassifierReady = false;
int eiLastTopLabelIndex = -1;
float eiLastTopLabelScore = 0.0f;
unsigned long eiLastResultPrintMs = 0;
float eiLastBackgroundScore = 0.0f;
float eiLastWaterScore = 0.0f;
uint32_t eiLastDspTimeMs = 0;
uint32_t eiLastClassificationTimeMs = 0;
uint32_t eiLastAnomalyTimeMs = 0;
bool eiHasResult = false;


// Serial output toggles
bool SERIAL_ASCII_SCOPE_ENABLED = false;
bool MIC_DIAGNOSTICS_ENABLED = false;
const unsigned long SERIAL_ASCII_SCOPE_INTERVAL_MS = 100;
const int32_t SERIAL_ASCII_SCOPE_RANGE = 20000;
const unsigned long MIC_DIAGNOSTICS_INTERVAL_MS = 1000;

// Mic processing config
const bool MIC_DC_BLOCK_ENABLED = true;
const int MIC_DC_FILTER_SHIFT = 7;   // Higher = slower DC tracking

// AGC config
const int AGC_TARGET_PEAK = 12000; // Target peak value after gain
const int AGC_MAX_GAIN = 16;
const int AGC_MIN_GAIN = 1;
const int AGC_ATTACK = 2;   // Higher = slower gain increase
const int AGC_DECAY = 6;    // Higher = slower gain decrease

int micSignalGain = 4; // Initial gain, will be adjusted by AGC

// UART pins for XIAO ESP32-S3
#define UART_TX 43  
#define UART_RX 44 

#define RECORD_SECONDS 5
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

#define PREBUFFER_SECONDS 3
#define PREBUFFER_SAMPLES (SAMPLE_RATE * PREBUFFER_SECONDS)

bool isRecording = false;
File recordingFile;
uint32_t samplesWritten = 0;
uint32_t recordingPrerollSamples = 0;
String activeRecordingPath = "";
unsigned long recordingStartMs = 0;
unsigned long recordingStopDeadlineMs = 0;
int32_t recordingPeakAbs = 0;

static SemaphoreHandle_t audioMutex = nullptr;
static TaskHandle_t audioCaptureTaskHandle = nullptr;

bool helloSent = false;
bool wifiConfigured = false;
bool webServerStarted = false;
unsigned long lastHelloMs = 0;
const unsigned long helloRetryIntervalMs = 2000;

int32_t* prebuffer = nullptr;

volatile uint32_t preIndex = 0;
unsigned long start = millis();
uint32_t ip = 0;
uint8_t status = 0;
unsigned long lastAsciiScopeMs = 0;
unsigned long lastMicDiagnosticsMs = 0;

uint32_t diagFrames = 0;
uint32_t diagSamples = 0;
uint32_t diagReadErrors = 0;
uint32_t diagShortReads = 0;
uint32_t diagNonZeroSamples = 0;
int32_t diagMinRaw = 32767;
int32_t diagMaxRaw = -32768;
uint64_t diagSumAbs = 0;
int64_t diagSum = 0;
int32_t diagMinProc = 32767;
int32_t diagMaxProc = -32768;
uint64_t diagSumAbsProc = 0;
int32_t micDcEstimate = 0;

struct MicPinCandidate {
    int bck;
    int ws;
    int din;
    const char* label;
};

HardwareSerial UART(1);
WebSocketsServer webSocket = WebSocketsServer(81); 
WebServer server(80); 

#define START_BYTE 0xAA

enum PacketType : uint8_t {
    HELLO = 0x01,
    REQUEST_CONFIG = 0x02,
    CONFIG_DATA = 0x03,
    NODE_STATUS = 0x04,
    HEARTBEAT = 0x05,
    ERROR_PKT = 0x06,
    CMD_RECORD_AUDIO = 0x10,
    CMD_PING         = 0x11,
    CMD_GET_STATUS   = 0x12,
    CMD_GET_SAMPLE   = 0x13,
    RESP_AUDIO_OK    = 0x14,
    RESP_PONG        = 0x15,
    RESP_STATUS      = 0x16,
    RESP_SAMPLE      = 0x17,
    CMD_RECORD_AUDIO_OFF = 0x18,
    CMD_GET_NODE_STATUS = 0x19,
    CMD_LED_PULSE_BLUE  = 0x1C,
    CMD_LED_STEADY_RED  = 0x1D,
    CMD_LED_HEALTHY     = 0x1E,
    CMD_FLED_PULSE      = 0x1F
};

struct WifiConfig {
    char ssid[32];
    char password[64];
    char apiBaseUrl[64];
    uint8_t deviceRole;
};

struct NodeStatus {
    uint8_t ip[4];
    uint8_t wifiStatus;
    uint8_t nodeRole;
};

// Basic health logging for overnight diagnostics.
static constexpr bool HEALTH_LOG_SERIAL = false;
static constexpr bool HEALTH_LOG_SD = true;
// Non-blocking floodlight timer: 0 = off, >0 = millis() deadline to turn off
static unsigned long fledOffAtMs = 0;

static constexpr unsigned long HEALTH_HEARTBEAT_INTERVAL_MS = 60000;
static constexpr const char* HEALTH_LOG_DIR = "/logs";
static constexpr const char* HEALTH_LOG_FALLBACK_FILE = "/logs/audio_health_current.log";
static constexpr size_t FILE_STREAM_CHUNK_SIZE = 1024;
static constexpr bool HEALTH_LOG_LIGHTWEIGHT_MODE = true;
static constexpr unsigned long HEALTH_AUDIO_STALL_THRESHOLD_MS = 10000;
static constexpr uint8_t HEALTH_AUDIO_STALL_CYCLES = 3;

// Recording reliability is the top priority for now.
static constexpr bool ENABLE_EDGE_IMPULSE_CLASSIFIER = true;
static constexpr bool ENABLE_MIC_PIN_DIAGNOSTIC = true;
static constexpr bool INCLUDE_PREROLL_IN_RECORDING = true;

static unsigned long lastHealthHeartbeatMs = 0;
static unsigned long lastUartPacketMs = 0;
static unsigned long lastWebSocketEventMs = 0;
static unsigned long lastWebRequestMs = 0;
static unsigned long lastLedStateChangeMs = 0;

static uint32_t healthUartRxCount = 0;
static uint32_t healthUartCrcFailCount = 0;
static uint32_t healthUartInvalidPayloadCount = 0;
static uint32_t healthWebSocketEventCount = 0;
static uint32_t healthWebRequestCount = 0;
static PacketType healthLastUartType = ERROR_PKT;
static bool healthLogStorageReady = false;
static String healthLogFilePath = HEALTH_LOG_FALLBACK_FILE;
static uint8_t fileStreamBuffer[FILE_STREAM_CHUNK_SIZE];
static portMUX_TYPE scopeFrameMux = portMUX_INITIALIZER_UNLOCKED;
static int16_t pendingScopeFrame[FRAME_SAMPLES] = {0};
static size_t pendingScopeFrameBytes = 0;
static volatile bool scopeFramePending = false;
static volatile unsigned long lastAudioFrameMs = 0;
static volatile uint32_t audioTaskLoopCount = 0;
static volatile uint32_t audioI2sErrCount = 0;
static volatile uint32_t audioWriteShortCount = 0;
static uint8_t healthAudioStallStreak = 0;
static bool healthAudioStallLatched = false;

static const char* packetTypeName(PacketType type);
static void logHealthEvent(const char* topic, const String& message);
static void ensureHealthLogStorage();
static void rotateHealthLogIfNeeded();
static String getHealthLogPathForDaysPast(int logDaysPast);
static bool streamTextFileResponse(const String& path);
static void applyCORSHeaders();
static void audioCaptureTask(void* parameter);
static void queueScopeFrame(const int16_t* samples, size_t byteCount);
static void broadcastPendingScopeFrame();
static void checkWifiWatchdog();

void sendPacket(HardwareSerial &port, PacketType type, const uint8_t *payload, uint8_t length) {
    uint8_t checksum = type ^ length;
    for (int i = 0; i < length; i++) checksum ^= payload[i];

    port.write(START_BYTE);
    port.write(type);
    port.write(length);
    port.write(payload, length);
    port.write(checksum);
}

bool readPacket(HardwareSerial &port, PacketType &typeOut, uint8_t *buffer, uint8_t &lengthOut) {
    static enum { WAIT_START, WAIT_TYPE, WAIT_LEN, WAIT_PAYLOAD, WAIT_CHECKSUM } state = WAIT_START;
    static uint8_t type, length, index, checksum;
    static unsigned long lastByteMs = 0;

    // If a frame stalls mid-parse, reset and wait for next START byte.
    if (state != WAIT_START && (millis() - lastByteMs) > 100) {
        logHealthEvent("UART", "Parser timeout reset.");
        state = WAIT_START;
    }

    while (port.available()) {
        uint8_t b = port.read();
        lastByteMs = millis();

        switch (state) {
            case WAIT_START:
                if (b == START_BYTE) state = WAIT_TYPE;
                break;

            case WAIT_TYPE:
                type = b;
                checksum = b;
                state = WAIT_LEN;
                break;

            case WAIT_LEN:
                length = b;
                checksum ^= b;
                index = 0;
                state = (length == 0) ? WAIT_CHECKSUM : WAIT_PAYLOAD;
                break;

            case WAIT_PAYLOAD:
                // Defensive guard in case of parser desync.
                if (index >= 256) {
                    state = WAIT_START;
                    break;
                }
                buffer[index++] = b;
                checksum ^= b;
                if (index >= length) state = WAIT_CHECKSUM;
                break;

            case WAIT_CHECKSUM:
                if (b == checksum) {
                    typeOut = (PacketType)type;
                    lengthOut = length;
                    state = WAIT_START;
                    return true;
                }
                healthUartCrcFailCount++;
                state = WAIT_START;
                break;
        }
    }
    return false;
}

extern const char* getDateTimeString() {
    static char dateTime[64];
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
        strftime(dateTime, sizeof(dateTime), "%Y-%m-%d %H:%M:%S", &timeinfo);
    } else {
        snprintf(dateTime, sizeof(dateTime), "MILLIS_%lu", millis());
    }
    return dateTime;
}

static const char* packetTypeName(PacketType type) {
    switch (type) {
        case HELLO: return "HELLO";
        case REQUEST_CONFIG: return "REQUEST_CONFIG";
        case CONFIG_DATA: return "CONFIG_DATA";
        case NODE_STATUS: return "NODE_STATUS";
        case HEARTBEAT: return "HEARTBEAT";
        case ERROR_PKT: return "ERROR_PKT";
        case CMD_RECORD_AUDIO: return "CMD_RECORD_AUDIO";
        case CMD_PING: return "CMD_PING";
        case CMD_GET_STATUS: return "CMD_GET_STATUS";
        case CMD_GET_SAMPLE: return "CMD_GET_SAMPLE";
        case RESP_AUDIO_OK: return "RESP_AUDIO_OK";
        case RESP_PONG: return "RESP_PONG";
        case RESP_STATUS: return "RESP_STATUS";
        case RESP_SAMPLE: return "RESP_SAMPLE";
        case CMD_RECORD_AUDIO_OFF: return "CMD_RECORD_AUDIO_OFF";
        case CMD_GET_NODE_STATUS: return "CMD_GET_NODE_STATUS";
        case CMD_LED_PULSE_BLUE: return "CMD_LED_PULSE_BLUE";
        case CMD_LED_STEADY_RED: return "CMD_LED_STEADY_RED";
        case CMD_LED_HEALTHY: return "CMD_LED_HEALTHY";
        case CMD_FLED_PULSE: return "CMD_FLED_PULSE";
        default: return "UNKNOWN";
    }
}

static void ensureHealthLogStorage() {
    if (!HEALTH_LOG_SD || healthLogStorageReady) {
        return;
    }

    if (!SD.exists(HEALTH_LOG_DIR)) {
        SD.mkdir(HEALTH_LOG_DIR);
    }

    // Create fallback file so logging always has a valid target.
    File f = SD.open(HEALTH_LOG_FALLBACK_FILE, FILE_APPEND);
    if (f) {
        f.close();
        healthLogStorageReady = true;
    }
}

static void rotateHealthLogIfNeeded() {
    if (!HEALTH_LOG_SD) {
        return;
    }

    ensureHealthLogStorage();

    struct tm timeinfo;
    String nextPath = HEALTH_LOG_FALLBACK_FILE;
    if (getLocalTime(&timeinfo)) {
        char dateTag[16];
        strftime(dateTag, sizeof(dateTag), "%Y-%m-%d", &timeinfo);
        nextPath = String(HEALTH_LOG_DIR) + "/audio_health_" + String(dateTag) + ".log";
    }

    if (nextPath != healthLogFilePath) {
        healthLogFilePath = nextPath;
    }

    // Touch file so it exists even before first write of a new day.
    File f = SD.open(healthLogFilePath, FILE_APPEND);
    if (f) {
        f.close();
    }
}

static String getHealthLogPathForDaysPast(int logDaysPast) {
    if (logDaysPast < 0) {
        logDaysPast = 0;
    }

    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
        timeinfo.tm_mday -= logDaysPast;
        mktime(&timeinfo);

        char dateTag[16];
        strftime(dateTag, sizeof(dateTag), "%Y-%m-%d", &timeinfo);
        return String(HEALTH_LOG_DIR) + "/audio_health_" + String(dateTag) + ".log";
    }

    // Fallback only makes sense for current day when RTC/time is unavailable.
    return (logDaysPast == 0) ? String(HEALTH_LOG_FALLBACK_FILE) : String("");
}

static bool streamTextFileResponse(const String& path) {
    if (!SD.exists(path)) {
        return false;
    }

    File f = SD.open(path, FILE_READ);
    if (!f) {
        return false;
    }

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.sendHeader("Content-Type", "text/plain");
    server.setContentLength(f.size());
    server.send(200, "text/plain", "");

    WiFiClient client = server.client();
    if (!client || !client.connected()) {
        f.close();
        return true;
    }

    while (f.available() && client.connected()) {
        size_t bytesRead = f.read(fileStreamBuffer, FILE_STREAM_CHUNK_SIZE);
        if (bytesRead == 0) {
            break;
        }

        size_t written = client.write(fileStreamBuffer, bytesRead);
        if (written != bytesRead) {
            break;
        }
    }

    f.close();
    return true;
}

static void queueScopeFrame(const int16_t* samples, size_t byteCount) {
    size_t clampedBytes = (byteCount > sizeof(pendingScopeFrame)) ? sizeof(pendingScopeFrame) : byteCount;
    taskENTER_CRITICAL(&scopeFrameMux);
    memcpy(pendingScopeFrame, samples, clampedBytes);
    pendingScopeFrameBytes = clampedBytes;
    scopeFramePending = (clampedBytes > 0);
    taskEXIT_CRITICAL(&scopeFrameMux);
}

static void broadcastPendingScopeFrame() {
    if (!scopeFramePending) {
        return;
    }

    int16_t localFrame[FRAME_SAMPLES] = {0};
    size_t localBytes = 0;

    taskENTER_CRITICAL(&scopeFrameMux);
    if (scopeFramePending) {
        localBytes = pendingScopeFrameBytes;
        memcpy(localFrame, pendingScopeFrame, localBytes);
        scopeFramePending = false;
        pendingScopeFrameBytes = 0;
    }
    taskEXIT_CRITICAL(&scopeFrameMux);

    if (localBytes == 0 || webSocket.connectedClients() == 0) {
        return;
    }

    webSocket.broadcastBIN((uint8_t*)localFrame, localBytes);
}

static void logHealthEvent(const char* topic, const String& message) {
    if (HEALTH_LOG_LIGHTWEIGHT_MODE) {
        bool allowed =
            (strcmp(topic, "HEALTH") == 0) ||
            (strcmp(topic, "BOOT") == 0) ||
            (strcmp(topic, "REC") == 0) ||
            (strcmp(topic, "WARN") == 0);
        if (!allowed) {
            return;
        }
    }

    String line = String("[HLOG ") + String(millis()) + "ms][" + topic + "] " + message;
    if (HEALTH_LOG_SERIAL) {
        Serial.println(line);
    }

    if (HEALTH_LOG_SD) {
        rotateHealthLogIfNeeded();
        File f = SD.open(healthLogFilePath, FILE_APPEND);
        if (f) {
            f.println(line);
            f.close();
        }
    }
}

void startRecording(String filen, uint32_t durationSeconds) {
    if (audioMutex == nullptr) return;

    if (xSemaphoreTake(audioMutex, portMAX_DELAY) != pdTRUE) {
        return;
    }
    bool alreadyRecording = isRecording;
    xSemaphoreGive(audioMutex);

    if (alreadyRecording) return;

    if (!SD.exists("/audio")) SD.mkdir("/audio");

    String filename = "/audio/" + filen + ".wav";
    filename.replace(":", "-");

    int16_t* prerollSnapshot = nullptr;
    if (INCLUDE_PREROLL_IN_RECORDING) {
        prerollSnapshot = (int16_t*) ps_malloc(PREBUFFER_SAMPLES * sizeof(int16_t));
        if (prerollSnapshot == nullptr) {
            Serial.println("Failed to allocate pre-roll snapshot");
            return;
        }

        if (xSemaphoreTake(audioMutex, portMAX_DELAY) == pdTRUE) {
            uint32_t idx = preIndex;
            for (int i = 0; i < PREBUFFER_SAMPLES; i++) {
                prerollSnapshot[i] = (int16_t)(prebuffer[idx] >> 16);
                idx = (idx + 1) % PREBUFFER_SAMPLES;
            }
            xSemaphoreGive(audioMutex);
        }
    }

    File file = SD.open(filename, FILE_WRITE);
    if (!file) {
        Serial.println("Failed to open WAV file");
        if (prerollSnapshot != nullptr) free(prerollSnapshot);
        return;
    }

    uint8_t blank[44] = {0};
    file.write(blank, 44);

    uint32_t prerollWritten = 0;
    int32_t prerollPeak = 0;

    if (prerollSnapshot != nullptr) {
        size_t bytesToWrite = PREBUFFER_SAMPLES * sizeof(int16_t);
        size_t bytesWritten = file.write((uint8_t*)prerollSnapshot, bytesToWrite);
        prerollWritten = bytesWritten / sizeof(int16_t);
        for (uint32_t i = 0; i < prerollWritten; i++) {
            int32_t absPcm = abs((int32_t)prerollSnapshot[i]);
            if (absPcm > prerollPeak) {
                prerollPeak = absPcm;
            }
        }
        free(prerollSnapshot);
    }

    if (xSemaphoreTake(audioMutex, portMAX_DELAY) != pdTRUE) {
        file.close();
        return;
    }

    samplesWritten = prerollWritten;
    recordingPrerollSamples = prerollWritten;
    recordingPeakAbs = prerollPeak;
    activeRecordingPath = filename;
    recordingStartMs = millis();
    recordingStopDeadlineMs = (durationSeconds == 0) ? 0 : (recordingStartMs + (durationSeconds * 1000UL));
    recordingFile = file;
    isRecording = true;

    xSemaphoreGive(audioMutex);

    logHealthEvent("REC", String("start file=") + filename +
                          " durSec=" + String(durationSeconds) +
                          " prerollSamples=" + String(recordingPrerollSamples));

    Serial.println(
        "Recording started: " + filename +
        " (for " + durationSeconds + " seconds" +
        (INCLUDE_PREROLL_IN_RECORDING ? ", +3s pre-roll" : "") + ")"
    );
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
        healthWebSocketEventCount++;
        lastWebSocketEventMs = millis();
    switch (type) {
      case WStype_DISCONNECTED:
                logHealthEvent("WS", String("client=") + num + " disconnected");
        break;
      case WStype_CONNECTED: {
        IPAddress ip = webSocket.remoteIP(num);
                logHealthEvent("WS", String("client=") + num + " connected ip=" + ip.toString());
        webSocket.sendTXT(num, "Connected");
      }
      break;
      case WStype_TEXT:
                logHealthEvent("WS", String("client=") + num + " text len=" + String(length));
        break;
    }
}

void setupOTA() {
    ElegantOTA.begin(&server);
    Serial.println("ElegantOTA ready");
}

static void applyCORSHeaders() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void setupWebServer() {
  server.on("/", []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        logHealthEvent("HTTP", "GET /");
        applyCORSHeaders();
    server.send(200, "text/plain", "ESP32-S3 Audio Node is running.");
  });
    
    server.on("/getAudioFileNames", HTTP_GET, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        logHealthEvent("HTTP", "GET /getAudioFileNames");
        applyCORSHeaders();
        String json = "[";
        if (SD.exists("/audio")) {
            File root = SD.open("/audio");
            if (root) {
                File file = root.openNextFile();
                bool first = true;
                while (file) {
                    if (!file.isDirectory()) {
                        String full = String(file.name());
                        int slash = full.lastIndexOf('/');
                        String fname = (slash >= 0) ? full.substring(slash + 1) : full;
                        String lname = fname;
                        lname.toLowerCase();
                        if (lname.endsWith(".wav")) {
                            if (!first) json += ",";
                            json += '"';
                            json += fname;
                            json += '"';
                            first = false;
                        }
                    }
                    file = root.openNextFile();
                }
                root.close();
            }
        }
        json += "]";
        server.send(200, "application/json", json);
    });

    server.on("/getAudioWAV", HTTP_GET, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        String filename = server.arg("filename");
        logHealthEvent("HTTP", String("GET /getAudioWAV file=") + (filename.isEmpty() ? "<missing>" : filename));
        if (filename.isEmpty()) {
            applyCORSHeaders();
            server.send(400, "text/plain", "Missing filename parameter");
            return;
        }

        String fullPath = "/audio/" + filename;

        // WAV headers are written when recording stops; block reads while file is active.
        if (isRecording && fullPath == activeRecordingPath) {
            applyCORSHeaders();
            server.send(409, "text/plain", "File is still being recorded; try again when recording stops");
            return;
        }

        if (!SD.exists(fullPath)) {
            applyCORSHeaders();
            server.send(404, "text/plain", "File not found: " + fullPath);
            return;
        }

        File audioFile = SD.open(fullPath, FILE_READ);
        if (!audioFile) {
            applyCORSHeaders();
            server.send(500, "text/plain", "Failed to open file");
            return;
        }

        applyCORSHeaders();
        size_t fileSize = audioFile.size();
        server.setContentLength(fileSize);
        server.send(200, "audio/wav", "");

        WiFiClient client = server.client();
        if (!client || !client.connected()) {
            audioFile.close();
            return;
        }

        while (audioFile.available() && client.connected()) {
            size_t bytesRead = audioFile.read(fileStreamBuffer, FILE_STREAM_CHUNK_SIZE);
            if (bytesRead == 0) {
                break;
            }

            size_t written = client.write(fileStreamBuffer, bytesRead);
            if (written != bytesRead) {
                break;
            }
        }

        audioFile.close();
    });

    server.on("/getAudioFileNames", HTTP_OPTIONS, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        applyCORSHeaders();
        server.send(200);
    });

    server.on("/getAudioWAV", HTTP_OPTIONS, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        applyCORSHeaders();
        server.send(200);
    });

    server.on("/getLog", HTTP_GET, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();

        String logType = server.arg("logType");
        if (logType.length() == 0) {
            logType = "AppLog";
        }

        int logDaysPast = 0;
        if (server.hasArg("logDaysPast")) {
            logDaysPast = server.arg("logDaysPast").toInt();
            if (logDaysPast < 0) {
                logDaysPast = 0;
            }
        }

        logHealthEvent("HTTP", String("GET /getLog logType=") + logType + " logDaysPast=" + String(logDaysPast));

        if (!logType.equalsIgnoreCase("AppLog")) {
            applyCORSHeaders();
            server.send(400, "text/plain", "Unsupported logType. Use logType=AppLog");
            return;
        }

        String logPath = getHealthLogPathForDaysPast(logDaysPast);
        if (logPath.length() == 0) {
            applyCORSHeaders();
            server.send(404, "text/plain", "Requested log day unavailable (time not set yet)");
            return;
        }

        if (!streamTextFileResponse(logPath)) {
            applyCORSHeaders();
            server.send(404, "text/plain", "Log file not found: " + logPath);
            return;
        }
    });

    server.on("/getLog", HTTP_OPTIONS, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        applyCORSHeaders();
        server.send(200);
    });

    server.on("/audioRecording", HTTP_GET, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        applyCORSHeaders();
        if (isRecording) {
            server.send(409, "application/json", "{\"status\":\"busy\",\"reason\":\"already recording\"}");
            return;
        }
        String label = server.hasArg("label") ? server.arg("label") : "motion";
        int duration = server.hasArg("duration") ? server.arg("duration").toInt() : 5;
        if (duration < 1) duration = 1;
        if (duration > 120) duration = 120;
        
        // Build filename: if DTS (device timestamp) provided, use it; otherwise use millis()
        // DTS allows audio file to be matched with video/photo files from main device
        String filename;
        if (server.hasArg("dts")) {
            filename = label + "_" + server.arg("dts");
        } else {
            filename = label + "_" + String(millis());
        }
        
        logHealthEvent("HTTP", String("GET /audioRecording label=") + label + " dur=" + String(duration) + " file=" + filename);
        startRecording(filename, duration);
        server.send(200, "application/json", "{\"status\":\"ok\",\"label\":\"" + label + "\",\"duration\":" + String(duration) + ",\"filename\":\"" + filename + "\"}");
    });

    server.on("/audioRecording", HTTP_OPTIONS, []() {
        healthWebRequestCount++;
        lastWebRequestMs = millis();
        applyCORSHeaders();
        server.send(200);
    });

  setupOTA();
  server.begin();
    logHealthEvent("HTTP", "Web server started");
  delay(100);
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

bool connectWifiAndStartServices(const char* ssid, const char* password, const char* sourceTag) {
    if (ssid == nullptr || ssid[0] == '\0') {
        return false;
    }

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);       // Keep radio always-on so inbound TCP connects are never missed
    WiFi.setAutoReconnect(true);
    WiFi.begin(ssid, password);

    Serial.printf("Connecting to WiFi via %s...\n", sourceTag);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        Serial.print(".");
        delay(250);
        attempts++;
    }

    wifiConfigured = (WiFi.status() == WL_CONNECTED);
    if (wifiConfigured) {
        Serial.println("\nWiFi connected.");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());

        if (!webServerStarted) {
            setupWebServer();
            webServerStarted = true;
        }
    } else {
        Serial.println("\nWiFi connect timeout.");
    }

    return wifiConfigured;
}

void printAsciiOscilloscope(int32_t minVal, int32_t maxVal) {
    int mid = WIDTH / 2;
    int minPos = map(minVal, -SERIAL_ASCII_SCOPE_RANGE, SERIAL_ASCII_SCOPE_RANGE, 0, WIDTH);
    int maxPos = map(maxVal, -SERIAL_ASCII_SCOPE_RANGE, SERIAL_ASCII_SCOPE_RANGE, 0, WIDTH);

    minPos = constrain(minPos, 0, WIDTH);
    maxPos = constrain(maxPos, 0, WIDTH);

    if (minPos > maxPos) {
        int swapPos = minPos;
        minPos = maxPos;
        maxPos = swapPos;
    }

    for (int i = 0; i < WIDTH; i++) {
        if (i == mid) {
            Serial.print('|');
        } else if (i >= minPos && i <= maxPos) {
            Serial.print('#');
        } else {
            Serial.print(' ');
        }
    }
    Serial.println();
}

static int eiGetAudioData(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)eiInferenceSlice[offset + i];
    }
    return 0;
}

static void runEdgeImpulseOnSlice() {
    if (!eiClassifierReady) {
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &eiGetAudioData;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, false);
    if (err != EI_IMPULSE_OK) {
        Serial.printf("EI run_classifier_continuous failed (%d)\n", err);
        return;
    }

    int topIndex = 0;
    float topValue = result.classification[0].value;
    for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > topValue) {
            topValue = result.classification[ix].value;
            topIndex = (int)ix;
        }
    }

    bool shouldPrint = false;
    if (topIndex != eiLastTopLabelIndex) {
        shouldPrint = true;
    }
    if (fabsf(topValue - eiLastTopLabelScore) >= 0.05f) {
        shouldPrint = true;
    }
    if ((millis() - eiLastResultPrintMs) >= EI_RESULT_PRINT_INTERVAL_MS) {
        shouldPrint = true;
    }

    if (shouldPrint) {
        Serial.printf(
            "EI: top=%s (%.3f) | Background=%.3f | Water(faucet)=%.3f | dsp=%lums cls=%lums anom=%lums\n",
            ei_classifier_inferencing_categories[topIndex],
            topValue,
            result.classification[0].value,
            result.classification[1].value,
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly
        );

        if (topIndex == 1 && topValue >= EI_INFERENCE_MIN_CONFIDENCE) {
            Serial.printf("EI event: Water(faucet) detected with confidence %.3f\n", topValue);
        }

        eiLastTopLabelIndex = topIndex;
        eiLastTopLabelScore = topValue;
        eiLastResultPrintMs = millis();
    }

    eiLastTopLabelIndex = topIndex;
    eiLastTopLabelScore = topValue;
    eiLastBackgroundScore = result.classification[0].value;
    eiLastWaterScore = result.classification[1].value;
    eiLastDspTimeMs = result.timing.dsp;
    eiLastClassificationTimeMs = result.timing.classification;
    eiLastAnomalyTimeMs = result.timing.anomaly;
    eiHasResult = true;
}

static void pushFrameToEdgeImpulse(const int16_t *samples, size_t sampleCount) {
    for (size_t i = 0; i < sampleCount; i++) {
        eiInferenceSlice[eiSliceWriteIndex++] = samples[i];

        if (eiSliceWriteIndex >= EI_CLASSIFIER_SLICE_SIZE) {
            runEdgeImpulseOnSlice();
            eiSliceWriteIndex = 0;
        }
    }
}

uint32_t runMicVarianceProbe(int sampleCount, int32_t &minOut, int32_t &maxOut) {
    int16_t s = 0;
    int16_t prev = 0;
    bool hasPrev = false;
    size_t bytesRead = 0;

    minOut = 32767;
    maxOut = -32768;
    uint32_t transitions = 0;

    for (int i = 0; i < sampleCount; i++) {
        esp_err_t err = i2s_read(I2S_NUM_0, &s, sizeof(s), &bytesRead, pdMS_TO_TICKS(20));
        if (err != ESP_OK || bytesRead != sizeof(s)) {
            continue;
        }

        int32_t v = (int32_t)s;
        if (v < minOut) minOut = v;
        if (v > maxOut) maxOut = v;

        if (hasPrev && s != prev) transitions++;
        prev = s;
        hasPrev = true;
    }

    return transitions;
}

void diagnoseAndSelectMicPins() {
    MicPinCandidate candidates[] = {
        {I2S_SCK, -1, I2S_SD, "PDM clk on BCK (current)"},
        {-1, I2S_SCK, I2S_SD, "PDM clk on WS (alt)"}
    };

    int32_t mins[2] = {0, 0};
    int32_t maxs[2] = {0, 0};
    int32_t spans[2] = {0, 0};
    uint32_t transitionsByCandidate[2] = {0, 0};
    int bestIndex = 0;
    int64_t bestScore = INT64_MIN;

    Serial.println("Running PDM pin-role diagnostic...");
    for (int i = 0; i < (int)(sizeof(candidates) / sizeof(candidates[0])); i++) {
        i2s_pin_config_t cfg = {
            .bck_io_num = candidates[i].bck,
            .ws_io_num = candidates[i].ws,
            .data_out_num = -1,
            .data_in_num = candidates[i].din
        };

        i2s_set_pin(I2S_NUM_0, &cfg);
        i2s_zero_dma_buffer(I2S_NUM_0);
        delay(30);

        int32_t pMin = 0;
        int32_t pMax = 0;
        uint32_t transitions = runMicVarianceProbe(1024, pMin, pMax);
        int32_t span = pMax - pMin;
        mins[i] = pMin;
        maxs[i] = pMax;
        spans[i] = span;
        transitionsByCandidate[i] = transitions;

        Serial.printf(
            "MIC probe [%s]: bck=%d ws=%d din=%d min=%ld max=%ld span=%ld transitions=%lu\n",
            candidates[i].label,
            candidates[i].bck,
            candidates[i].ws,
            candidates[i].din,
            (long)pMin,
            (long)pMax,
            (long)span,
            (unsigned long)transitions
        );

        // Score candidate by real activity first; penalize clipped/static streams.
        bool clippedLow = (pMin <= -32760 && pMax <= 0);
        bool clippedHigh = (pMax >= 32760 && pMin >= 0);
        bool likelyStatic = (transitions < 32);

        int64_t score = (int64_t)transitions * 1000LL + (int64_t)span;
        if ((clippedLow || clippedHigh) && likelyStatic) {
            score -= 1000000LL;
        }

        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    if (spans[bestIndex] < 100 || transitionsByCandidate[bestIndex] < 32) {
        logHealthEvent("WARN", String("MIC probe low activity spans=") + spans[0] + "/" + spans[1] +
                              " transitions=" + transitionsByCandidate[0] + "/" + transitionsByCandidate[1]);
    }

    i2s_pin_config_t selected = {
        .bck_io_num = candidates[bestIndex].bck,
        .ws_io_num = candidates[bestIndex].ws,
        .data_out_num = -1,
        .data_in_num = candidates[bestIndex].din
    };
    i2s_set_pin(I2S_NUM_0, &selected);
    i2s_zero_dma_buffer(I2S_NUM_0);

    Serial.printf(
        "Selected MIC mapping: %s (bck=%d ws=%d din=%d)\n",
        candidates[bestIndex].label,
        candidates[bestIndex].bck,
        candidates[bestIndex].ws,
        candidates[bestIndex].din
    );
}

void writeWavHeader(File &file, int dataSize, int sampleRate) {
    uint8_t header[44];

    memcpy(header, "RIFF", 4);
    uint32_t fileSize = dataSize + 36;
    memcpy(header + 4, &fileSize, 4);
    memcpy(header + 8, "WAVEfmt ", 8);

    uint32_t fmtChunkSize = 16;
    memcpy(header + 16, &fmtChunkSize, 4);

    uint16_t audioFormat = 1;
    memcpy(header + 20, &audioFormat, 2);

    uint16_t numChannels = 1;
    memcpy(header + 22, &numChannels, 2);

    memcpy(header + 24, &sampleRate, 4);

    uint32_t byteRate = sampleRate * numChannels * 2;
    memcpy(header + 28, &byteRate, 4);

    uint16_t blockAlign = numChannels * 2;
    memcpy(header + 32, &blockAlign, 2);

    uint16_t bitsPerSample = 16;
    memcpy(header + 34, &bitsPerSample, 2);

    memcpy(header + 36, "data", 4);
    memcpy(header + 40, &dataSize, 4);

    file.write(header, 44);
}

void saveWavToSD(String timestamp, int32_t *buffer, int totalSamples) {
    if (!SD.exists("/audio")) SD.mkdir("/audio");

    String filename = "/audio/audio_" + timestamp + ".wav";
    filename.replace(":", "-");

    File f = SD.open(filename, FILE_WRITE);
    if (!f) {
        Serial.println("Failed to open WAV file");
        return;
    }

    writeWavHeader(f, totalSamples * sizeof(int16_t), SAMPLE_RATE);

    for (int i = 0; i < totalSamples; i++) {
        int16_t pcm = (int16_t)(buffer[i] >> 16);
        f.write((uint8_t*)&pcm, sizeof(int16_t));
    }

    f.close();
    Serial.println("Saved WAV: " + filename);
}

void recordAudioSnippet(String timestamp) {
    Serial.println("Triggered recording...");

    const int totalSamples = PREBUFFER_SAMPLES + (SAMPLE_RATE * 5);

    int32_t* fullBuffer = (int32_t*) ps_calloc(totalSamples, sizeof(int32_t));
    if (!fullBuffer) {
        Serial.println("ERROR: Could not allocate PSRAM buffer!");
        return;
    }

    int32_t* ptr = fullBuffer;

    uint32_t idx = preIndex;
    for (int i = 0; i < PREBUFFER_SAMPLES; i++) {
        *ptr++ = prebuffer[idx];
        idx = (idx + 1) % PREBUFFER_SAMPLES;
    }

    size_t bytes_read;
    for (int i = 0; i < SAMPLE_RATE * 5; i++) {
        int16_t sample;
        i2s_read(I2S_NUM_0, &sample, sizeof(sample), &bytes_read, portMAX_DELAY);
        *ptr++ = ((int32_t)sample) << 16;
    }

    saveWavToSD(timestamp, fullBuffer, totalSamples);

    free(fullBuffer);

    Serial.println("Recording complete.");
}

#define MCP23017_ADDR 0x27  // I2C address for MCP23017 IO expander
const int MCP_INT_A_PIN = 1;
const int MCP_INT_B_PIN = 2;
MCP23017 mcp(MCP23017_ADDR);  // global MCP23017 instance

// ── Software PWM LED control via MCP23017 ────────────────────────────────────
// Brightness: 0 (off) to 7 (full on), per R/G/B channel.
// PWM period = 8 ticks × 12 ms ≈ 96 ms (~10 Hz) — adequate for status LEDs.

static constexpr uint8_t MCP_LED_RED_PIN   = STATUS_LED_MCP_RED_PIN;
static constexpr uint8_t MCP_LED_GREEN_PIN = STATUS_LED_MCP_GREEN_PIN;
static constexpr uint8_t MCP_LED_BLUE_PIN  = STATUS_LED_MCP_BLUE_PIN;
static constexpr bool    LED_ACTIVE_LOW    = (STATUS_LED_ACTIVE_LOW != 0);

static inline uint8_t ledLevel(bool on) {
    return (on != LED_ACTIVE_LOW) ? HIGH : LOW;
}

struct LedColor { uint8_t r, g, b; };   // each channel 0–7
static volatile LedColor ledTarget = {0, 0, 0};

enum LedAnimationMode : uint8_t {
    LED_MODE_HEALTHY_PULSE = 0,
    LED_MODE_MANUAL = 1,
};

static volatile LedAnimationMode ledAnimationMode = LED_MODE_HEALTHY_PULSE;
static volatile uint32_t healthyPulseCycleStartMs = 0;
static volatile LedColor ledPulseBase = {0, 7, 0};
// Shadow of Port B output latch — avoids read-modify-write per tick.
// HIGH bit = off for active-low LEDs; init all high (all off).
static uint8_t portBShadow = 0xFF;
static SemaphoreHandle_t i2cMutex  = nullptr;

static const char* ledModeName(LedAnimationMode mode) {
    return (mode == LED_MODE_HEALTHY_PULSE) ? "pulse" : "manual";
}

static void logHealthHeartbeatIfDue() {
    unsigned long now = millis();
    if ((now - lastHealthHeartbeatMs) < HEALTH_HEARTBEAT_INTERVAL_MS) {
        return;
    }

    // Avoid extra SD work while an audio recording is active.
    if (isRecording) {
        return;
    }

    lastHealthHeartbeatMs = now;

    unsigned long audioMsAgo = (lastAudioFrameMs == 0) ? 0 : (now - lastAudioFrameMs);

    if (audioMsAgo > HEALTH_AUDIO_STALL_THRESHOLD_MS) {
        if (healthAudioStallStreak < 255) {
            healthAudioStallStreak++;
        }
        if (healthAudioStallStreak >= HEALTH_AUDIO_STALL_CYCLES) {
            if (!healthAudioStallLatched || ((healthAudioStallStreak % 5) == 0)) {
                logHealthEvent(
                    "WARN",
                    String("audio task stale msAgo=") + String(audioMsAgo) +
                    " streak=" + String(healthAudioStallStreak) +
                    " loops=" + String(audioTaskLoopCount) +
                    " i2sErr=" + String(audioI2sErrCount)
                );
            }
            healthAudioStallLatched = true;
        }
    } else {
        if (healthAudioStallLatched) {
            logHealthEvent("WARN", String("audio task recovered msAgo=") + String(audioMsAgo));
        }
        healthAudioStallStreak = 0;
        healthAudioStallLatched = false;
    }

    String msg = String("heap=") + ESP.getFreeHeap() +
                 " minHeap=" + ESP.getMinFreeHeap() +
                 " wifi=" + (WiFi.status() == WL_CONNECTED ? "up" : "down") +
                 " rec=" + (isRecording ? "on" : "off") +
                 " wsClients=" + webSocket.connectedClients() +
                 " audioMsAgo=" + String(audioMsAgo) +
                 " audioLoops=" + String(audioTaskLoopCount) +
                 " i2sErr=" + String(audioI2sErrCount) +
                 " writeShort=" + String(audioWriteShortCount) +
                 " ledMode=" + ledModeName(ledAnimationMode) +
                 " pulseBase=[" + String(ledPulseBase.r) + "," + String(ledPulseBase.g) + "," + String(ledPulseBase.b) + "]" +
                 " ledTarget=[" + String(ledTarget.r) + "," + String(ledTarget.g) + "," + String(ledTarget.b) + "]" +
                 " uartRx=" + String(healthUartRxCount) +
                 " uartCrcFail=" + String(healthUartCrcFailCount) +
                 " uartInvalid=" + String(healthUartInvalidPayloadCount) +
                 " lastUartMsAgo=" + String((lastUartPacketMs == 0) ? 0 : (now - lastUartPacketMs)) +
                 " lastUartType=" + packetTypeName(healthLastUartType) +
                 " httpReq=" + String(healthWebRequestCount) +
                 " wsEv=" + String(healthWebSocketEventCount) +
                 " lastWsMsAgo=" + String((lastWebSocketEventMs == 0) ? 0 : (now - lastWebSocketEventMs)) +
                 " lastHttpMsAgo=" + String((lastWebRequestMs == 0) ? 0 : (now - lastWebRequestMs)) +
                 " lastLedChangeMsAgo=" + String((lastLedStateChangeMs == 0) ? 0 : (now - lastLedStateChangeMs));

    logHealthEvent("HEALTH", msg);
}

static void setLedAnimationMode(LedAnimationMode mode) {
    LedAnimationMode oldMode = ledAnimationMode;
    ledAnimationMode = mode;
    if (mode == LED_MODE_HEALTHY_PULSE) {
        healthyPulseCycleStartMs = millis();
    }
    if (oldMode != mode) {
        lastLedStateChangeMs = millis();
        logHealthEvent("LED", String("mode ") + ledModeName(oldMode) + " -> " + ledModeName(mode));
    }
}

static void setPulseBaseColor(uint8_t r, uint8_t g, uint8_t b) {
    ledPulseBase.r = (r > 7) ? 7 : r;
    ledPulseBase.g = (g > 7) ? 7 : g;
    ledPulseBase.b = (b > 7) ? 7 : b;
}

static void setHealthyGreenPulse() {
    setPulseBaseColor(0, 7, 0);
    setLedAnimationMode(LED_MODE_HEALTHY_PULSE);
    lastLedStateChangeMs = millis();
    logHealthEvent("LED", "healthy green pulse");
}

static void setBluePulse() {
    setPulseBaseColor(0, 0, 7);
    setLedAnimationMode(LED_MODE_HEALTHY_PULSE);
    lastLedStateChangeMs = millis();
    logHealthEvent("LED", "pulse blue");
}

static void setSteadyRed() {
    setLedAnimationMode(LED_MODE_MANUAL);
    setLedColor(7, 0, 0);
    lastLedStateChangeMs = millis();
    logHealthEvent("LED", "steady red");
}

static void pulseFloodLightMs(unsigned long onMs) {
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        mcp.pinMode(1, OUTPUT);      // MCP A1
        mcp.digitalWrite(1, HIGH);   // Floodlight ON
        xSemaphoreGive(i2cMutex);
    }

    delay(onMs);

    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        mcp.digitalWrite(1, LOW);    // Floodlight OFF
        xSemaphoreGive(i2cMutex);
    }
}

// Set the current LED color. Values are clamped to 0–7.
void setLedColor(uint8_t r, uint8_t g, uint8_t b) {
    uint8_t rIn = (r > 7) ? 7 : r;
    uint8_t gIn = (g > 7) ? 7 : g;
    uint8_t bIn = (b > 7) ? 7 : b;

    // Apply simple per-channel scaling for color balancing.
    ledTarget.r = (uint8_t)((rIn * STATUS_LED_RED_MAX   + 3) / 7);
    ledTarget.g = (uint8_t)((gIn * STATUS_LED_GREEN_MAX + 3) / 7);
    ledTarget.b = (uint8_t)((bIn * STATUS_LED_BLUE_MAX  + 3) / 7);
}

// FreeRTOS task: soft-PWM loop, 8 ticks × 2 ms per cycle ≈ 62 Hz — above flicker fusion.
// One OLAT_B register write per tick (single I2C transaction) keeps the bus load low.
static void ledPwmTask(void*) {
    // Pre-compute bit masks from the pin defines (Port B: pin - 8 = bit index)
    static constexpr uint8_t R_BIT = 1u << (MCP_LED_RED_PIN   - 8);  // bit 2
    static constexpr uint8_t G_BIT = 1u << (MCP_LED_GREEN_PIN - 8);  // bit 6
    static constexpr uint8_t B_BIT = 1u << (MCP_LED_BLUE_PIN  - 8);  // bit 7
    uint8_t tick = 0;
    for (;;) {
        uint8_t r = ledTarget.r, g = ledTarget.g, b = ledTarget.b;
        bool rOn = (tick < r), gOn = (tick < g), bOn = (tick < b);
        // Build Port B latch byte from shadow; set/clear each LED bit.
        uint8_t pb = portBShadow;
        if (ledLevel(rOn)) pb |= R_BIT; else pb &= ~R_BIT;
        if (ledLevel(gOn)) pb |= G_BIT; else pb &= ~G_BIT;
        if (ledLevel(bOn)) pb |= B_BIT; else pb &= ~B_BIT;
        portBShadow = pb;
        if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            mcp.writeRegister(MCP23017Register::OLAT_B, pb);
            xSemaphoreGive(i2cMutex);
        }
        tick = (tick + 1) & 0x07;  // wrap 0–7
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}

// Background status animation: persistent healthy pulse (green)
// Cadence mirrors demo timing and can be overridden by LED_MODE_MANUAL.
static void ledHealthyPulseTask(void*) {
    static constexpr uint32_t FADE_IN_MS = 1500;
    static constexpr uint32_t HOLD_MS = 2500;
    static constexpr uint32_t FADE_OUT_MS = 1500;
    static constexpr uint32_t OFF_MS = 1000;
    static constexpr uint32_t CYCLE_MS = FADE_IN_MS + HOLD_MS + FADE_OUT_MS + OFF_MS;

    healthyPulseCycleStartMs = millis();

    for (;;) {
        if (ledAnimationMode == LED_MODE_HEALTHY_PULSE) {
            uint32_t elapsed = millis() - healthyPulseCycleStartMs;
            uint32_t phase = elapsed % CYCLE_MS;

            uint8_t level = 0;
            if (phase < FADE_IN_MS) {
                level = (uint8_t)((7u * phase + (FADE_IN_MS / 2u)) / FADE_IN_MS);
            } else if (phase < (FADE_IN_MS + HOLD_MS)) {
                level = 7;
            } else if (phase < (FADE_IN_MS + HOLD_MS + FADE_OUT_MS)) {
                uint32_t down = phase - (FADE_IN_MS + HOLD_MS);
                level = (uint8_t)((7u * (FADE_OUT_MS - down) + (FADE_OUT_MS / 2u)) / FADE_OUT_MS);
            } else {
                level = 0;
            }

            uint8_t baseR = ledPulseBase.r;
            uint8_t baseG = ledPulseBase.g;
            uint8_t baseB = ledPulseBase.b;
            uint8_t outR = (uint8_t)((baseR * level + 3u) / 7u);
            uint8_t outG = (uint8_t)((baseG * level + 3u) / 7u);
            uint8_t outB = (uint8_t)((baseB * level + 3u) / 7u);
            setLedColor(outR, outG, outB);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// Fade LED color: fade-in → hold at full → fade-out (off)
// Uses linear interpolation for smooth brightness transitions
static void fadeLedColor(uint8_t targetR, uint8_t targetG, uint8_t targetB, 
                         unsigned long fadeInMs, unsigned long holdMs, unsigned long fadeOutMs) {
    unsigned long startTime = millis();
    unsigned long fadeInEnd = startTime + fadeInMs;
    unsigned long holdEnd = fadeInEnd + holdMs;
    unsigned long fadeOutEnd = holdEnd + fadeOutMs;
    
    unsigned long currentTime;
    while ((currentTime = millis()) < fadeOutEnd) {
        if (currentTime < fadeInEnd) {
            // Fade-in phase: interpolate from 0 to target brightness
            float progress = (float)(currentTime - startTime) / fadeInMs;
            if (progress > 1.0f) progress = 1.0f;
            
            uint8_t r = (uint8_t)(targetR * progress);
            uint8_t g = (uint8_t)(targetG * progress);
            uint8_t b = (uint8_t)(targetB * progress);
            
            setLedColor(r, g, b);
        } else if (currentTime < holdEnd) {
            // Hold phase: maintain full brightness
            setLedColor(targetR, targetG, targetB);
        } else {
            // Fade-out phase: interpolate from target down to 0 (off)
            float progress = (float)(currentTime - holdEnd) / fadeOutMs;
            if (progress > 1.0f) progress = 1.0f;
            
            uint8_t r = (uint8_t)(targetR * (1.0f - progress));
            uint8_t g = (uint8_t)(targetG * (1.0f - progress));
            uint8_t b = (uint8_t)(targetB * (1.0f - progress));
            
            setLedColor(r, g, b);
        }
        
        // Small delay to avoid busy-waiting; 20ms ≈ 50 Hz update rate for smooth fade
        delay(20);
    }
    
    // Ensure clean end state (all off)
    setLedColor(0, 0, 0);
}

// 9-color sequence with fade-in/hold/fade-out effect (~5.5 seconds per color)
static void ledColorDemo() {
    setLedAnimationMode(LED_MODE_MANUAL);

    struct { uint8_t r, g, b; const char* name; } palette[] = {
        {7, 0, 0, "Red"},
        {0, 7, 0, "Green"},
        {0, 0, 7, "Blue"},
        {7, 7, 0, "Yellow"},
        {0, 7, 7, "Cyan"},
        {7, 0, 7, "Magenta"},
        {7, 3, 0, "Orange"},
        {7, 7, 7, "White"},
        {3, 3, 3, "White (dim)"},
    };
    
    // Timing configuration (in milliseconds) — total cycle ~5.5s per color
    const unsigned long fadeInMs = 1500;   // 1.5 seconds fade-in to full
    const unsigned long holdMs = 2500;     // 2.5 seconds hold at full brightness
    const unsigned long fadeOutMs = 1500;  // 1.5 seconds fade-out to off
    
    for (auto& c : palette) {
        Serial.printf("LED demo: %-12s [%d, %d, %d] — fading in...\n", c.name, c.r, c.g, c.b);
        fadeLedColor(c.r, c.g, c.b, fadeInMs, holdMs, fadeOutMs);
    }

    setLedColor(0, 0, 0);
    setHealthyGreenPulse();
    Serial.println("LED demo complete — returning to healthy green pulse.");
}

void initStatusLedPwm() {
    // Initialise pins to output, all off
    mcp.pinMode(MCP_LED_RED_PIN,   OUTPUT);
    mcp.pinMode(MCP_LED_GREEN_PIN, OUTPUT);
    mcp.pinMode(MCP_LED_BLUE_PIN,  OUTPUT);
    // Write initial all-off state to OLAT_B in one transaction
    portBShadow = 0xFF;  // active-low: HIGH = off
    mcp.writeRegister(MCP23017Register::OLAT_B, portBShadow);
    // Create I2C mutex and launch PWM task
    i2cMutex = xSemaphoreCreateMutex();
    xTaskCreate(ledPwmTask, "ledPwm", 3072, nullptr, 1, nullptr);
    xTaskCreate(ledHealthyPulseTask, "ledHealthy", 2048, nullptr, 1, nullptr);
    setHealthyGreenPulse();
    Serial.println("LED PWM + healthy pulse tasks started.");
}

void setupMCP23017() {
    Serial.println("Initializing MCP23017 (Lemasle)...");
    mcp.init();
    uint8_t iocon = mcp.readRegister(MCP23017Register::IOCON);
    iocon |= 0b00000100;   // set ODR bit
    mcp.writeRegister(MCP23017Register::IOCON, iocon);
    mcp.readRegister(MCP23017Register::INTCAP_A);
    mcp.readRegister(MCP23017Register::INTCAP_B);
    mcp.writeRegister(MCP23017Register::IODIR_A, 0xFF);
    mcp.writeRegister(MCP23017Register::IODIR_B, 0xFF);
    mcp.writeRegister(MCP23017Register::GPPU_A, 0xFF);
    mcp.writeRegister(MCP23017Register::GPPU_B, 0xFF);
    pinMode(MCP_INT_A_PIN, INPUT_PULLUP);
    pinMode(MCP_INT_B_PIN, INPUT_PULLUP);
    Serial.println("MCP23017 initialization complete.");
    mcp.pinMode(0, OUTPUT);
    mcp.pinMode(1, OUTPUT);
   //mcp.digitalWrite(1, HIGH);
    //delay(1000);
    //mcp.digitalWrite(1, LOW);
}

void setup() {
  Serial.begin(115200);
  delay(10000);

  SPI.begin(7, 8, 9, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("Card Mount Failed");
  }
  else {
    Serial.println("Card Initialized");
        ensureHealthLogStorage();
        rotateHealthLogIfNeeded();
  }

    delay(5000);
  UART.begin(115200, SERIAL_8N1, UART_RX, UART_TX);
  Serial.println("UART initialized on pins RX=" + String(UART_RX) + " TX=" + String(UART_TX));

    Serial.println("Waiting for WiFi credentials via UART CONFIG_DATA.");

    
  delay(15000);

    uint8_t helloPayload[1] = { 0x01 };
    sendPacket(UART, HELLO, helloPayload, 1);
    helloSent = true;
    lastHelloMs = millis();

  pinMode(5, INPUT);
  pinMode(BUZZER, OUTPUT);

  //digitalWrite(BUZZER, HIGH);
  // delay (1000);
  digitalWrite(BUZZER, LOW);
  
  prebuffer = (int32_t*) ps_calloc(PREBUFFER_SAMPLES, sizeof(int32_t));
  if (!prebuffer) {
      Serial.println("FATAL: Could not allocate prebuffer in PSRAM");
      while (1) delay(100);
  }

  // PDM Microphone I2S configuration
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = -1,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);

    if (ENABLE_MIC_PIN_DIAGNOSTIC) {
        diagnoseAndSelectMicPins();
    }

    Serial.printf("PDM MIC pins: CLK=GPIO%d DATA=GPIO%d\n", I2S_SCK, I2S_SD);

  delay(5);

  Wire.begin(5, 6);  // SDA=GPIO5, SCL=GPIO6
  setupMCP23017();
  initStatusLedPwm();
  //ledColorDemo();

    audioMutex = xSemaphoreCreateMutex();
    if (audioMutex == nullptr) {
        Serial.println("FATAL: Could not create audio mutex");
        while (1) delay(100);
    }

    xTaskCreatePinnedToCore(audioCaptureTask, "audioCapture", 8192, nullptr, 2, &audioCaptureTaskHandle, 1);

    if (ENABLE_EDGE_IMPULSE_CLASSIFIER) {
        run_classifier_init();
        eiClassifierReady = true;
        Serial.printf(
                "Edge Impulse ready: labels=%d, slice=%d samples, interval=%.2fms\n",
                EI_CLASSIFIER_LABEL_COUNT,
                EI_CLASSIFIER_SLICE_SIZE,
                EI_CLASSIFIER_INTERVAL_MS * EI_CLASSIFIER_SLICE_SIZE
        );
    } else {
        eiClassifierReady = false;
        Serial.println("Edge Impulse classifier disabled (recording-priority mode)");
    }

  Serial.println("S3 Audio Node Ready (PDM Microphone + UART)");
  UART.println("BOOT OK");
    logHealthEvent("BOOT", "AudioNode ready; health logging active");
}



void stopRecording() {
    if (audioMutex == nullptr) return;

    File fileToClose;
    String recordingPath;
    uint32_t totalSamples = 0;
    uint32_t prerollSamples = 0;
    int32_t peakAbs = 0;
    unsigned long elapsedMs = 0;

    if (xSemaphoreTake(audioMutex, portMAX_DELAY) != pdTRUE) {
        return;
    }

    if (!isRecording) {
        xSemaphoreGive(audioMutex);
        return;
    }

    fileToClose = recordingFile;
    recordingPath = activeRecordingPath;
    totalSamples = samplesWritten;
    prerollSamples = recordingPrerollSamples;
    peakAbs = recordingPeakAbs;
    elapsedMs = millis() - recordingStartMs;

    isRecording = false;
    samplesWritten = 0;
    activeRecordingPath = "";
    recordingStartMs = 0;
    recordingStopDeadlineMs = 0;
    recordingPrerollSamples = 0;
    recordingPeakAbs = 0;

    xSemaphoreGive(audioMutex);

    uint32_t postTriggerSamples = (totalSamples > prerollSamples)
        ? (totalSamples - prerollSamples)
        : 0;

    Serial.printf(
        "Recording stats: total=%lu preroll=%lu post=%lu elapsedMs=%lu headerRate=%lu peak=%ld\n",
        (unsigned long)totalSamples,
        (unsigned long)prerollSamples,
        (unsigned long)postTriggerSamples,
        (unsigned long)elapsedMs,
        (unsigned long)SAMPLE_RATE,
        (long)peakAbs
    );

    if (peakAbs < 400) {
        logHealthEvent("WARN", String("recording near-silent peak=") + String(peakAbs) +
                              " total=" + String(totalSamples) +
                              " post=" + String(postTriggerSamples));
    }

    fileToClose.seek(0);
    writeWavHeader(fileToClose, totalSamples * sizeof(int16_t), SAMPLE_RATE);

    fileToClose.close();

    logHealthEvent("REC", String("stop file=") + recordingPath +
                          " total=" + String(totalSamples) +
                          " post=" + String(postTriggerSamples) +
                          " elapsedMs=" + String(elapsedMs) +
                          " peak=" + String(peakAbs));

    Serial.println("Recording stopped.");
}

static void audioCaptureTask(void* parameter) {
    (void)parameter;

    int16_t frame[FRAME_SAMPLES];

    for (;;) {
        size_t bytesRead = 0;
        esp_err_t readErr = i2s_read(I2S_NUM_0, frame, sizeof(frame), &bytesRead, portMAX_DELAY);
        if (readErr != ESP_OK) {
            diagReadErrors++;
            audioI2sErrCount++;
            continue;
        }

        size_t validSamples = bytesRead / sizeof(int16_t);
        if (validSamples == 0 || validSamples > FRAME_SAMPLES) {
            diagShortReads++;
            continue;
        }

        lastAudioFrameMs = millis();
        audioTaskLoopCount++;

        int32_t minVal = 999999;
        int32_t maxVal = -999999;
        int32_t framePeak = 0;

        for (size_t i = 0; i < validSamples; i++) {
            int32_t sRaw = (int32_t)frame[i];

            diagSamples++;
            if (sRaw != 0) diagNonZeroSamples++;
            if (sRaw < diagMinRaw) diagMinRaw = sRaw;
            if (sRaw > diagMaxRaw) diagMaxRaw = sRaw;
            diagSumAbs += abs(sRaw);
            diagSum += sRaw;

            int32_t sProc = sRaw;
            if (MIC_DC_BLOCK_ENABLED) {
                micDcEstimate += (sRaw - micDcEstimate) >> MIC_DC_FILTER_SHIFT;
                sProc = sRaw - micDcEstimate;
            }
            sProc *= micSignalGain;
            sProc = constrain(sProc, -32768, 32767);

            if (sProc < diagMinProc) diagMinProc = sProc;
            if (sProc > diagMaxProc) diagMaxProc = sProc;
            diagSumAbsProc += abs(sProc);

            if (sProc < minVal) minVal = sProc;
            if (sProc > maxVal) maxVal = sProc;

            int32_t absProc = abs(sProc);
            if (absProc > framePeak) framePeak = absProc;

            prebuffer[preIndex] = sProc << 16;
            preIndex = (preIndex + 1) % PREBUFFER_SAMPLES;
            frame[i] = (int16_t)sProc;
        }

        diagFrames++;

        if (framePeak > 0) {
            if (framePeak > AGC_TARGET_PEAK && micSignalGain > AGC_MIN_GAIN) {
                micSignalGain -= (micSignalGain + AGC_ATTACK - 1) / AGC_ATTACK;
                if (micSignalGain < AGC_MIN_GAIN) micSignalGain = AGC_MIN_GAIN;
            } else if (framePeak < AGC_TARGET_PEAK / 2 && micSignalGain < AGC_MAX_GAIN) {
                micSignalGain += (micSignalGain + AGC_DECAY - 1) / AGC_DECAY;
                if (micSignalGain > AGC_MAX_GAIN) micSignalGain = AGC_MAX_GAIN;
            }
        }

        if (audioMutex != nullptr && xSemaphoreTake(audioMutex, portMAX_DELAY) == pdTRUE) {
            if (isRecording) {
                size_t bytesToWrite = validSamples * sizeof(int16_t);
                size_t bytesWritten = recordingFile.write((uint8_t*)frame, bytesToWrite);
                size_t writtenSamples = bytesWritten / sizeof(int16_t);
                if (bytesWritten != bytesToWrite) {
                    audioWriteShortCount++;
                }
                samplesWritten += (uint32_t)writtenSamples;
                for (size_t i = 0; i < writtenSamples; i++) {
                    int32_t absPcm = abs((int32_t)frame[i]);
                    if (absPcm > recordingPeakAbs) {
                        recordingPeakAbs = absPcm;
                    }
                }
            }
            xSemaphoreGive(audioMutex);
        }

        if (ENABLE_EDGE_IMPULSE_CLASSIFIER && !isRecording) {
            pushFrameToEdgeImpulse(frame, validSamples);
        }

        // Keep oscilloscope streaming active during idle operation, but mute while recording.
        if (!isRecording) {
            int16_t scopeFrame[FRAME_SAMPLES] = {0};
            for (size_t i = 0; i < validSamples; i++) {
                int32_t scaled = (int32_t)frame[i] * 8;
                scopeFrame[i] = (int16_t)constrain(scaled, -32768, 32767);
            }
            queueScopeFrame(scopeFrame, sizeof(scopeFrame));
        }

        int32_t center = (minVal + maxVal) / 2;
        minVal -= center;
        maxVal -= center;
        minVal /= 2;
        maxVal /= 2;

        int32_t peak = max(abs(minVal), abs(maxVal));
        const int32_t NOISE_FLOOR = 1000;
        if (peak < NOISE_FLOOR) {
            minVal = 0;
            maxVal = 0;
        }

        if (SERIAL_ASCII_SCOPE_ENABLED && (millis() - lastAsciiScopeMs) >= SERIAL_ASCII_SCOPE_INTERVAL_MS) {
            printAsciiOscilloscope(minVal, maxVal);
            lastAsciiScopeMs = millis();
        }

        if (MIC_DIAGNOSTICS_ENABLED && (millis() - lastMicDiagnosticsMs) >= MIC_DIAGNOSTICS_INTERVAL_MS) {
            if (diagSamples > 0) {
                float avgAbs = (float)diagSumAbs / (float)diagSamples;
                float mean = (float)diagSum / (float)diagSamples;
                float avgAbsProc = (float)diagSumAbsProc / (float)diagSamples;
                Serial.printf(
                    "MIC diag: frames=%lu samples=%lu nz=%lu raw[min=%ld max=%ld avgAbs=%.1f mean=%.1f] proc[min=%ld max=%ld avgAbs=%.1f] dc=%ld gain=%d readErr=%lu short=%lu\n",
                    (unsigned long)diagFrames,
                    (unsigned long)diagSamples,
                    (unsigned long)diagNonZeroSamples,
                    (long)diagMinRaw,
                    (long)diagMaxRaw,
                    avgAbs,
                    mean,
                    (long)diagMinProc,
                    (long)diagMaxProc,
                    avgAbsProc,
                    (long)micDcEstimate,
                    micSignalGain,
                    (unsigned long)diagReadErrors,
                    (unsigned long)diagShortReads
                );
            } else {
                Serial.printf(
                    "MIC diag: no samples read; gain=%d readErr=%lu short=%lu\n",
                    micSignalGain,
                    (unsigned long)diagReadErrors,
                    (unsigned long)diagShortReads
                );
            }

            diagFrames = 0;
            diagSamples = 0;
            diagReadErrors = 0;
            diagShortReads = 0;
            diagNonZeroSamples = 0;
            diagMinRaw = 32767;
            diagMaxRaw = -32768;
            diagSumAbs = 0;
            diagSum = 0;
            diagMinProc = 32767;
            diagMaxProc = -32768;
            diagSumAbsProc = 0;
            lastMicDiagnosticsMs = millis();
        }
    }
}

void handlePacket(PacketType type, uint8_t *buffer, uint8_t len) {
    switch (type) {
      case CMD_RECORD_AUDIO: {
                if (len == 0) {
            healthUartInvalidPayloadCount++;
                        Serial.println("CMD_RECORD_AUDIO ignored: empty payload.");
                        break;
                }
        String incoming = String((char*)buffer).substring(0, len);

        int p1 = incoming.indexOf('|');
        int p2 = incoming.indexOf('|', p1 + 1);

        String timestamp = incoming.substring(0, p1);
        String label     = incoming.substring(p1 + 1, p2);
        String duration  = incoming.substring(p2 + 1);

        int durationSec = duration.toInt();
        String filen = timestamp+"."+label;

        Serial.println("CMD_RECORD_AUDIO received with timestamp: " + filen);
        startRecording(filen, durationSec);
        sendPacket(UART, RESP_AUDIO_OK, (uint8_t*)filen.c_str(), filen.length());
        break;
      }

      case CMD_RECORD_AUDIO_OFF:
          if (len != 0) {
              healthUartInvalidPayloadCount++;
              Serial.println("CMD_RECORD_AUDIO_OFF ignored: unexpected payload.");
              break;
          }
          stopRecording();
          sendPacket(UART, RESP_AUDIO_OK, nullptr, 0);
          break;

        case CMD_PING:
            if (len != 0) {
                healthUartInvalidPayloadCount++;
                Serial.println("CMD_PING ignored: unexpected payload.");
                break;
            }
            sendPacket(UART, RESP_PONG, nullptr, 0);
            break;

        case CMD_GET_NODE_STATUS: {
                    if (len != 0) {
                            logHealthEvent("UART", String("CMD_GET_NODE_STATUS received with extra payload len=") + String(len) + "; ignoring payload and replying with NODE_STATUS");
          }
          NodeStatus st;
          IPAddress ip = WiFi.localIP();
          st.ip[0] = ip[0];
          st.ip[1] = ip[1];
          st.ip[2] = ip[2];
          st.ip[3] = ip[3];
          st.wifiStatus = wifiConfigured ? 1 : 0;
          st.nodeRole = 1;

          sendPacket(UART, NODE_STATUS, (uint8_t*)&st, sizeof(st));
            break;
        }            

        case CMD_GET_STATUS: {
            String mssg = "STATUS NORMAL";
            if (eiHasResult && eiLastTopLabelIndex >= 0) {
                mssg += "|top=";
                mssg += ei_classifier_inferencing_categories[eiLastTopLabelIndex];
                mssg += " (";
                mssg += String(eiLastTopLabelScore, 3);
                mssg += ")|Background=";
                mssg += String(eiLastBackgroundScore, 3);
                mssg += "|Water(faucet)=";
                mssg += String(eiLastWaterScore, 3);
                mssg += "|dsp=";
                mssg += String(eiLastDspTimeMs);
                mssg += "ms|cls=";
                mssg += String(eiLastClassificationTimeMs);
                mssg += "ms|anom=";
                mssg += String(eiLastAnomalyTimeMs);
                mssg += "ms";
            } else {
                mssg += "|ei=pending";
            }

            uint8_t payloadLen = (mssg.length() > 255) ? 255 : (uint8_t)mssg.length();
            sendPacket(UART, RESP_STATUS, (uint8_t*)mssg.c_str(), payloadLen);
            break;
        }

        case CMD_GET_SAMPLE: {
            if (len < 8) {
                healthUartInvalidPayloadCount++;
                Serial.println("CMD_GET_SAMPLE ignored: payload too short.");
                break;
            }
            int32_t minVal = *((int32_t*)buffer);
            int32_t maxVal = *((int32_t*)(buffer+4));
            int32_t reply[2] = {minVal, maxVal};
            sendPacket(UART, RESP_SAMPLE, (uint8_t*)reply, sizeof(reply));
            break;
        }

        case CMD_LED_PULSE_BLUE:
            if (len != 0) {
                healthUartInvalidPayloadCount++;
                Serial.println("CMD_LED_PULSE_BLUE ignored: unexpected payload.");
                break;
            }
            setBluePulse();
            Serial.println("LED command: pulse blue (pump active).");
            break;

        case CMD_LED_STEADY_RED:
            if (len != 0) {
                healthUartInvalidPayloadCount++;
                Serial.println("CMD_LED_STEADY_RED ignored: unexpected payload.");
                break;
            }
            setSteadyRed();
            Serial.println("LED command: steady red (alarm active).");
            break;

        case CMD_LED_HEALTHY:
            if (len != 0) {
                healthUartInvalidPayloadCount++;
                Serial.println("CMD_LED_HEALTHY ignored: unexpected payload.");
                break;
            }
            setHealthyGreenPulse();

            
            Serial.println("LED command: healthy green pulse.");
            break;

            case CMD_FLED_PULSE: {
                // Payload: optional 4-byte big-endian uint32 duration in ms. Default 5000 ms.
                // Special case: 0 ms means immediate OFF and timer cancel.
                unsigned long onMs = 5000;
                if (len >= 4) {
                    onMs = ((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16)
                         | ((uint32_t)buffer[2] << 8)  |  (uint32_t)buffer[3];
                }

                if (onMs == 0) {
                    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                        mcp.pinMode(1, OUTPUT);
                        mcp.digitalWrite(1, LOW);
                        xSemaphoreGive(i2cMutex);
                    }
                    fledOffAtMs = 0;
                    Serial.println("[FLED] Floodlight OFF (immediate command).");
                    logHealthEvent("FLED", "off immediate");
                    break;
                }

                if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                    mcp.pinMode(1, OUTPUT);
                    mcp.digitalWrite(1, HIGH);
                    xSemaphoreGive(i2cMutex);
                }
                fledOffAtMs = millis() + onMs;
                Serial.printf("[FLED] Floodlight ON for %lu ms.\n", onMs);
                logHealthEvent("FLED", String("on for ") + String(onMs) + "ms");
                break;
            }

        case CONFIG_DATA: {
                    if (len != sizeof(WifiConfig)) {
                    healthUartInvalidPayloadCount++;
                            Serial.printf("CONFIG_DATA ignored: invalid payload length %u (expected %u).\n", len, (unsigned)sizeof(WifiConfig));
                            break;
                    }
          WifiConfig cfg;
          memcpy(&cfg, buffer, sizeof(cfg));

          helloSent = true;
          lastHelloMs = millis();

          Serial.println("CONFIG_DATA received");
          Serial.printf("SSID: %s\n", cfg.ssid);
          Serial.printf("Password: %s\n", cfg.password);

          bool connected = connectWifiAndStartServices(cfg.ssid, cfg.password, "UART CONFIG_DATA");
          if (!connected) {
              Serial.println("WiFi connect timeout. Waiting for next config handshake...");
              helloSent = false;
          }

          NodeStatus st;
          IPAddress ip = WiFi.localIP();
          st.ip[0] = ip[0];
          st.ip[1] = ip[1];
          st.ip[2] = ip[2];
          st.ip[3] = ip[3];
          st.wifiStatus = wifiConfigured ? 1 : 0;
          st.nodeRole = 1;

          sendPacket(UART, NODE_STATUS, (uint8_t*)&st, sizeof(st));
          break;
      }
    }
}

// WiFi connectivity watchdog.
//
// PRIMARY fix: reconnect detection.
// When WiFi drops and auto-reconnects, the lwIP stack silently tears down all TCP
// server sockets. server.handleClient() and webSocket.loop() continue to run but
// accept nothing. Tracking the previous WiFi connection state lets us detect the
// transition and immediately restart the listening sockets.
//
// BACKSTOP: if no inbound TCP activity is seen for 20 min while WiFi appears up,
// force a reconnect cycle to clear any degraded radio/stack state. This covers
// edge cases that don't produce a visible disconnect event.
static void checkWifiWatchdog() {
    if (!wifiConfigured) return;

    // --- Reconnect detection ---
    static bool prevWifiUp = false;
    bool wifiUp = (WiFi.status() == WL_CONNECTED);

    if (!prevWifiUp && wifiUp && webServerStarted) {
        // WiFi just came back up. Restart TCP services so the server sockets are
        // re-bound to the new network interface.
        webSocket.close();
        server.stop();
        server.begin();
        webSocket.begin();
        webSocket.onEvent(webSocketEvent);
        // Advance timestamps so the 20-min idle backstop doesn't immediately trigger.
        unsigned long now = millis();
        lastWebRequestMs     = now;
        lastWebSocketEventMs = now;
        logHealthEvent("WIFI", "reconnect detected; TCP services restarted");
    }
    prevWifiUp = wifiUp;

    if (!wifiUp || isRecording) return;

    // --- 20-min idle backstop ---
    static constexpr unsigned long IDLE_THRESHOLD_MS = 20UL * 60UL * 1000UL;
    static unsigned long lastTriggerMs = 0;
    unsigned long now = millis();
    unsigned long lastActivity = max(lastWebRequestMs, lastWebSocketEventMs);
    unsigned long idleMs = (lastActivity == 0) ? now : (now - lastActivity);

    if (idleMs >= IDLE_THRESHOLD_MS &&
        (lastTriggerMs == 0 || now - lastTriggerMs > IDLE_THRESHOLD_MS)) {
        logHealthEvent("WIFI", String("watchdog: ") + String(idleMs / 60000) + "min idle; forcing reconnect");
        lastTriggerMs        = now;
        lastWebRequestMs     = now;  // prevent re-trigger while reconnect is in flight
        lastWebSocketEventMs = now;
        WiFi.reconnect();            // triggers disconnect→reconnect; detection block handles service restart
    }
}

void loop() {
        // --- Serial commands for local testing (rec, led, and UART LED packet simulation) ---
        static String serialCmd = "";
        uint8_t serialDummyPayload[1] = {0};
        while (Serial.available()) {
            char c = Serial.read();
            if (c == '\n' || c == '\r') {
                if (serialCmd.equalsIgnoreCase("rec")) {
                    String ts = String(getDateTimeString());
                    Serial.println("[SERIAL] Triggering 10s recording to SD card...");
                    startRecording(ts, 10);
                } else if (serialCmd.equalsIgnoreCase("led")) {
                    Serial.println("[SERIAL] Running LED color demo...");
                    ledColorDemo();
                } else if (serialCmd.equalsIgnoreCase("CMD_LED_PULSE_BLUE") || serialCmd.equalsIgnoreCase("blue")) {
                    Serial.println("[SERIAL] Simulating UART packet: CMD_LED_PULSE_BLUE");
                    handlePacket(CMD_LED_PULSE_BLUE, serialDummyPayload, 0);
                } else if (serialCmd.equalsIgnoreCase("CMD_LED_STEADY_RED") || serialCmd.equalsIgnoreCase("red")) {
                    Serial.println("[SERIAL] Simulating UART packet: CMD_LED_STEADY_RED");
                    handlePacket(CMD_LED_STEADY_RED, serialDummyPayload, 0);
                } else if (serialCmd.equalsIgnoreCase("CMD_LED_HEALTHY") || serialCmd.equalsIgnoreCase("green") || serialCmd.equalsIgnoreCase("healthy")) {
                    Serial.println("[SERIAL] Simulating UART packet: CMD_LED_HEALTHY");
                    handlePacket(CMD_LED_HEALTHY, serialDummyPayload, 0);
                } else if (serialCmd.equalsIgnoreCase("FLED")) {
                    Serial.println("[SERIAL] Floodlight test: ON for 5 seconds...");
                    pulseFloodLightMs(5000);
                    Serial.println("[SERIAL] Floodlight test complete.");
                } else if (serialCmd.length() > 0) {
                    Serial.println("[SERIAL] Unknown command. Try: rec, led, CMD_LED_PULSE_BLUE, CMD_LED_STEADY_RED, CMD_LED_HEALTHY, FLED");
                }
                serialCmd = "";
            } else if (isPrintable(c)) {
                serialCmd += c;
                if (serialCmd.length() > 48) serialCmd = ""; // prevent runaway
            }
        }
    PacketType type;
    uint8_t buffer[256];
    uint8_t len;

    if (!wifiConfigured && (!helloSent || (millis() - lastHelloMs) >= helloRetryIntervalMs)) {
        uint8_t helloPayload[1] = { 0x01 };
        sendPacket(UART, HELLO, helloPayload, 1);
        Serial.println("Sent HELLO packet, waiting for config...");
        helloSent = true;
        lastHelloMs = millis();
    }

    if (readPacket(UART, type, buffer, len)) {
        healthUartRxCount++;
        healthLastUartType = type;
        lastUartPacketMs = millis();
        handlePacket(type, buffer, len);
    }

    logHealthHeartbeatIfDue();

    if (!isRecording) {
        webSocket.loop();
        broadcastPendingScopeFrame();
        ElegantOTA.loop();
        server.handleClient();
        checkWifiWatchdog();
    }
     
    if (isRecording && recordingStopDeadlineMs > 0 && millis() >= recordingStopDeadlineMs) {
        stopRecording();
    }

        // Non-blocking floodlight turn-off
        if (fledOffAtMs > 0 && millis() >= fledOffAtMs) {
            if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                mcp.digitalWrite(1, LOW);
                xSemaphoreGive(i2cMutex);
            }
            fledOffAtMs = 0;
            Serial.println("[FLED] Floodlight OFF.");
            logHealthEvent("FLED", "off");
        }

    delay(10);
}