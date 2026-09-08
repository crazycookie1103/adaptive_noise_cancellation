// components/tinyml_control/include/tinyml_inference.h
//
// Wraps the TFLite Micro interpreter for the int8-quantized DS-CNN
// classifier exported by export_tflite.py / tinyml_anc_model.h.
//
// Depends on the `esp-tflite-micro` component (Espressif's TFLite Micro
// port for ESP-IDF). Add it via:
//   idf.py add-dependency "espressif/esp-tflite-micro"

#pragma once

#include <cstdint>

enum class TinyMLClass : int {
    kStationary = 0,
    kNonStationary = 1,
    kSpeech = 2,
};

struct TinyMLResult {
    TinyMLClass cls;
    float confidence;   // dequantized softmax value of the argmax class
    bool valid;          // false if inference failed (arena OOM, etc.)
};

class TinyMLInference {
public:
    TinyMLInference();
    ~TinyMLInference();

    // Allocates the interpreter + tensor arena. Call once at startup,
    // after TinyMLFeatureExtractor::init(). Returns false on failure
    // (check logs -- almost always a tensor-arena-too-small error, bump
    // kTensorArenaSize if so).
    bool init();

    // Runs one inference given a (TINYML_N_FRAMES x TINYML_N_MELS)
    // float32 feature map (already normalized -- see
    // TinyMLFeatureExtractor::compute_features). Handles quantization
    // to int8 internally.
    TinyMLResult classify(const float *features, size_t n_features);

private:
    // Sized generously for this ~10 KB model; TFLite Micro's arena also
    // holds intermediate activation tensors, not just weights. Tune down
    // once you've measured actual usage via
    // interpreter_->arena_used_bytes().
    static constexpr int kTensorArenaSize = 32 * 1024;
    uint8_t *tensor_arena_ = nullptr;

    void *interpreter_ = nullptr;  // opaque: tflite::MicroInterpreter*
    bool initialized_ = false;
};

const char *tinyml_class_to_string(TinyMLClass cls);
