// main/tinyml_control_task.cpp
//
// The async TinyML control loop, running as its own FreeRTOS task at
// ~100-200 ms cadence, per the architecture diagram. This is separate
// from (and lower priority than) the real-time sample-by-sample NLMS/
// Wiener audio loop -- it only ever WRITES the current suppression
// params, which the audio loop reads.
//
// Wire-up assumptions (adjust to your actual audio capture setup):
//   - `audio_ring_buffer_read()` pulls the newest TINYML_HOP_SAMPLES of
//     int16 mic audio from the primary channel, AFTER Stage 0's
//     kurtosis impulse filter has already run on it.
//   - `g_control_state` is read by the real-time audio task via
//     ControlState::current_params() -- guard with a mutex or use
//     std::atomic<float> per field if the two tasks run on different
//     cores (ESP32-S3 is dual-core); a plain struct read/write across
//     cores without synchronization is a real bug waiting to happen.

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "tinyml_features.h"
#include "tinyml_inference.h"
#include "control_interface.h"

static const char *TAG = "tinyml_task";

static TinyMLFeatureExtractor g_feature_extractor;
static TinyMLInference g_inference;
static ControlState g_control_state(/*step_smooth=*/0.2f);

// TODO: replace with your actual mic ring-buffer read, returning the
// newest TINYML_HOP_SAMPLES samples (post Stage-0 impulse filter).
extern size_t audio_ring_buffer_read(int16_t *out, size_t max_samples);

// Called from the real-time audio loop (different task/core) to fetch
// the latest smoothed params. Wrap with a mutex if needed for your RTOS
// config -- left as a plain accessor here since locking strategy depends
// on how your existing NLMS/Wiener loop is structured.
SuppressionParams tinyml_get_current_params() {
    return g_control_state.current_params();
}

static void tinyml_control_task(void *pvParameters) {
    static float features[TINYML_N_FRAMES * TINYML_N_MELS];
    int16_t hop_buf[TINYML_HOP_SAMPLES];

    ESP_LOGI(TAG, "TinyML control task started (hop=%d ms, window=%d ms)",
              TINYML_HOP_MS, TINYML_WINDOW_MS);

    const TickType_t period = pdMS_TO_TICKS(TINYML_HOP_MS);
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        size_t n = audio_ring_buffer_read(hop_buf, TINYML_HOP_SAMPLES);
        if (n > 0) {
            g_feature_extractor.push_audio(hop_buf, n);
            g_feature_extractor.compute_features(features);

            TinyMLResult result = g_inference.classify(
                features, TINYML_N_FRAMES * TINYML_N_MELS);

            if (result.valid) {
                g_control_state.update(result);
                ESP_LOGD(TAG, "class=%s confidence=%.2f mu=%.2f floor_gain=%.2f",
                          tinyml_class_to_string(result.cls), result.confidence,
                          g_control_state.current_params().mu,
                          g_control_state.current_params().floor_gain);
            }
        }

        // Fixed-period wake regardless of how long classification took,
        // matching the "runs every 100-200 ms" async cadence -- if
        // inference occasionally overruns 100ms, this naturally degrades
        // toward the 200ms end of that range rather than drifting.
        vTaskDelayUntil(&last_wake, period);
    }
}

void tinyml_control_init(void) {
    if (!g_feature_extractor.init()) {
        ESP_LOGE(TAG, "Feature extractor init failed");
        return;
    }
    if (!g_inference.init()) {
        ESP_LOGE(TAG, "TFLite Micro inference init failed");
        return;
    }

    // Lower priority than the real-time audio task; pin to the core NOT
    // running the sample-by-sample DSP loop if you're using both cores.
    xTaskCreatePinnedToCore(
        tinyml_control_task, "tinyml_ctrl", /*stack=*/8192,
        nullptr, /*priority=*/tskIDLE_PRIORITY + 3, nullptr, /*core=*/1);
}
