// components/tinyml_control/tinyml_features.cpp

#include "tinyml_features.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "esp_dsp.h"   // dsps_fft2r_fc32, dsps_bit_rev2r_fc32, dsps_cplx2real_fc32
#include "esp_log.h"

static const char *TAG = "tinyml_features";

TinyMLFeatureExtractor::TinyMLFeatureExtractor() {
    memset(audio_buffer_, 0, sizeof(audio_buffer_));
}

TinyMLFeatureExtractor::~TinyMLFeatureExtractor() {}

bool TinyMLFeatureExtractor::init() {
    esp_err_t err = dsps_fft2r_init_fc32(NULL, TINYML_N_FFT);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %d", err);
        return false;
    }
    fft_initialized_ = true;
    return true;
}

void TinyMLFeatureExtractor::push_audio(const int16_t *samples, size_t n) {
    // Shift the rolling buffer left by n, append new samples converted
    // to float in [-1, 1] -- mirrors features.py's fixed WINDOW_SAMPLES
    // buffer semantics (np.roll + assign the tail).
    if (n >= TINYML_WINDOW_SAMPLES) {
        // More new audio than the whole window: just take the tail.
        const int16_t *tail = samples + (n - TINYML_WINDOW_SAMPLES);
        for (size_t i = 0; i < TINYML_WINDOW_SAMPLES; ++i) {
            audio_buffer_[i] = tail[i] / 32768.0f;
        }
        return;
    }

    size_t keep = TINYML_WINDOW_SAMPLES - n;
    memmove(audio_buffer_, audio_buffer_ + n, keep * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        audio_buffer_[keep + i] = samples[i] / 32768.0f;
    }
}

void TinyMLFeatureExtractor::compute_features(float *out_features) {
    for (int frame = 0; frame < TINYML_N_FRAMES; ++frame) {
        int start = frame * TINYML_FRAME_STEP;

        // Windowed frame, zero-padded to N_FFT, packed as interleaved
        // complex (imag = 0) for esp-dsp's in-place complex FFT.
        memset(fft_buf_, 0, sizeof(fft_buf_));
        for (int i = 0; i < TINYML_FRAME_LEN; ++i) {
            fft_buf_[2 * i] = audio_buffer_[start + i] * g_hann_window[i];
            fft_buf_[2 * i + 1] = 0.0f;
        }

        dsps_fft2r_fc32(fft_buf_, TINYML_N_FFT);
        dsps_bit_rev2r_fc32(fft_buf_, TINYML_N_FFT);

        // Power spectrum for bins [0, N_FFT/2], matching
        // np.fft.rfft(...).real**2 + imag**2, normalized by N_FFT
        // (features.py: power = (re^2 + im^2) / N_FFT).
        float power[TINYML_N_FFT_BINS];
        for (int k = 0; k < TINYML_N_FFT_BINS; ++k) {
            float re = fft_buf_[2 * k];
            float im = fft_buf_[2 * k + 1];
            power[k] = (re * re + im * im) / TINYML_N_FFT;
        }

        // Mel filterbank -> log energies, one row of the output.
        float *out_row = out_features + frame * TINYML_N_MELS;
        for (int m = 0; m < TINYML_N_MELS; ++m) {
            float energy = 0.0f;
            for (int k = 0; k < TINYML_N_FFT_BINS; ++k) {
                energy += g_mel_filterbank[m][k] * power[k];
            }
            out_row[m] = logf(energy + 1e-6f);
        }
    }

    // Per-utterance mean/std normalization, matching features.py exactly.
    const int total = TINYML_N_FRAMES * TINYML_N_MELS;
    float mean = 0.0f;
    for (int i = 0; i < total; ++i) mean += out_features[i];
    mean /= total;

    float var = 0.0f;
    for (int i = 0; i < total; ++i) {
        float d = out_features[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / total) + 1e-6f;

    for (int i = 0; i < total; ++i) {
        out_features[i] = (out_features[i] - mean) / std;
    }
}
