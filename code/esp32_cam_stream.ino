/*
  ESP32-CAM MJPEG streamer for Chess Vision Recorder.
  Board: AI Thinker ESP32-CAM
  Endpoint: http://<esp32-ip>:81/stream

  v2 — fixes corrupt-frame transmission (FFmpeg `error dc` / `overread` at y=59),
       DMA race at last MCU row, and adds WiFi watchdog.
*/

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// -----------------------------
// Update these for your network
// -----------------------------
const char* WIFI_SSID = "ARRIS-409C";
const char* WIFI_PASSWORD = "DCA6333D409C";
const bool FLIP_VERTICAL   = true; // true = flip image upside-down
const bool FLIP_HORIZONTAL = false; // true = correct a mirrored image
const int  TARGET_FPS      = 15;   // target stream framerate (cap, not floor)

// AI Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

static httpd_handle_t stream_httpd = NULL;

// A valid JPEG must start with SOI (FF D8) and end with EOI (FF D9).
// Frames that fail this check are partially-DMA'd and would surface as
// `error dc` / `overread` in the FFmpeg MJPEG decoder.
static inline bool jpeg_valid(const uint8_t* buf, size_t len) {
  return (len > 4) &&
         (buf[0] == 0xFF) && (buf[1] == 0xD8) &&
         (buf[len - 2] == 0xFF) && (buf[len - 1] == 0xD9);
}

static esp_err_t index_handler(httpd_req_t* req) {
  const char* msg =
    "ESP32-CAM Chess Vision Stream\n"
    "Use: /stream\n";
  return httpd_resp_send(req, msg, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t stream_handler(httpd_req_t* req) {
  camera_fb_t* fb = NULL;
  esp_err_t res = ESP_OK;
  char header_buf[128];  // safe margin for large frames (100KB+ size = 6 digits)

  static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* _STREAM_HEADER_FMT =
    "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  const TickType_t frame_ticks = pdMS_TO_TICKS(1000 / TARGET_FPS);
  TickType_t last_wake = xTaskGetTickCount();
  uint32_t frame_count   = 0;
  uint32_t corrupt_count = 0;
  uint32_t fps_start_ms  = millis();

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    // Drop frames that didn't finish DMA. Sending a truncated JPEG produces
    // `error dc` / `overread` at the final MCU row in the downstream decoder.
    if (!jpeg_valid(fb->buf, fb->len)) {
      esp_camera_fb_return(fb);
      fb = NULL;
      corrupt_count++;
      vTaskDelayUntil(&last_wake, frame_ticks);
      continue;
    }

    // single combined boundary+header chunk halves TCP write overhead
    size_t hlen = snprintf(header_buf, sizeof(header_buf), _STREAM_HEADER_FMT, fb->len);
    res = httpd_resp_send_chunk(req, header_buf, hlen);
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);
    fb = NULL;
    if (res != ESP_OK) {
      break;
    }

    if (++frame_count >= 30) {
      uint32_t elapsed = millis() - fps_start_ms;
      Serial.printf("Stream: %.1f fps | corrupt dropped: %u\n",
                    (frame_count * 1000.0f) / elapsed, corrupt_count);
      frame_count   = 0;
      corrupt_count = 0;
      fps_start_ms  = millis();
    }

    // cap at TARGET_FPS but don't block if encoding already took longer
    vTaskDelayUntil(&last_wake, frame_ticks);
  }

  return res;
}

void start_camera_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 8;
  config.send_wait_timeout = 10;   // seconds; default 5s drops frames under WiFi jitter

  httpd_uri_t index_uri = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = index_handler,
    .user_ctx = NULL
  };

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void setup_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;        // 20MHz: 24MHz is OV2640's rated max and races DMA at last MCU row
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always serve newest frame, drop stale ones
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;    // 640x480 — sweet spot for 15fps over WiFi
    config.jpeg_quality = 10;             // higher quality = more robust coefficient stream
    config.fb_count = 3;                  // extra buffer absorbs WiFi send jitter (PSRAM is plentiful)
  } else {
    config.frame_size = FRAMESIZE_QVGA;   // 320x240 — no PSRAM means tiny buffers
    config.jpeg_quality = 10;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    while (true) {
      delay(1000);
    }
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != NULL) {
    sensor->set_vflip(sensor, FLIP_VERTICAL   ? 1 : 0);
    sensor->set_hmirror(sensor, FLIP_HORIZONTAL ? 1 : 0);

    // Stabilize AGC/AWB so inter-frame variance doesn't spike encode size
    sensor->set_framesize(sensor, FRAMESIZE_VGA);
    sensor->set_quality(sensor, 10);
    sensor->set_gainceiling(sensor, GAINCEILING_2X);
    sensor->set_exposure_ctrl(sensor, 1);
    sensor->set_awb_gain(sensor, 1);
    sensor->set_whitebal(sensor, 1);

    // Chess board tuning: sharper edges, less color noise
    sensor->set_contrast(sensor,    1);
    sensor->set_saturation(sensor, -1);
    sensor->set_sharpness(sensor,   1);
  }
}

void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                       // huge fps win: disables 802.11 power-save latency
  WiFi.setTxPower(WIFI_POWER_19_5dBm);        // max TX power for stable throughput
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. IP: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  setup_camera();
  setup_wifi();
  start_camera_server();

  Serial.print("Stream ready at: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
}

void loop() {
  // WiFi watchdog: reconnect on link loss, hard-restart if reconnect fails
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost — reconnecting...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
      delay(500);
      Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("\nReconnected. IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
      Serial.println("\nReconnect failed — restarting.");
      ESP.restart();
    }
  }
  delay(5000);
}

