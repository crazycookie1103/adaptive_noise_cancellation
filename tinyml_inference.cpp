// components/tinyml_control/tinyml_inference.cpp

#include "tinyml_inference.h"
#include "tinyml_anc_model.h"   // g_tinyml_anc_model[], g_tinyml_anc_model_len

#include <cstring>

#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "tinyml_inference";

// Ops used by model.py's DS-CNN: standard conv, depthwise conv, batchnorm
// (folded into conv at conversion time), relu, avg pool, softmax, reshape.
// Keep this list in sync if you change the architecture in model.py --
// MicroMutableOpResolver only registers what you list, deliberately, to
// keep the binary small.
namespace {
tflite::MicroMutableOpResolver<8> g_resolver;
const tflite::Model *g_model = nullptr;
}  // namespace

TinyMLInference::TinyMLInference() {}

TinyMLInference::~TinyMLInference() {
    if (tensor_arena_) {
        delete[] tensor_arena_;
    }
    if (interpreter_) {
        delete static_cast<tflite::MicroInterpreter *>(interpreter_);
    }
}

bool TinyMLInference::init() {
    g_model = tflite::GetModel(g_tinyml_anc_model);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: %lu != %d",
                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    if (g_resolver.AddConv2D() != kTfLiteOk ||
        g_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        g_resolver.AddRelu() != kTfLiteOk ||
        g_resolver.AddMean() != kTfLiteOk ||          // GlobalAveragePooling2D lowers to Mean
        g_resolver.AddFullyConnected() != kTfLiteOk ||
        g_resolver.AddSoftmax() != kTfLiteOk ||
        g_resolver.AddReshape() != kTfLiteOk ||
        g_resolver.AddQuantize() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to register one or more ops");
        return false;
    }

    tensor_arena_ = new (std::nothrow) uint8_t[kTensorArenaSize];
    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate %d byte tensor arena", kTensorArenaSize);
        return false;
    }

    auto *interpreter = new tflite::MicroInterpreter(
        g_model, g_resolver, tensor_arena_, kTensorArenaSize);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed -- try increasing kTensorArenaSize "
                      "(currently %d bytes)", kTensorArenaSize);
        delete interpreter;
        return false;
    }

    ESP_LOGI(TAG, "Tensor arena used: %u / %d bytes",
             (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);

    interpreter_ = interpreter;
    initialized_ = true;
    return true;
}

TinyMLResult TinyMLInference::classify(const float *features, size_t n_features) {
    TinyMLResult result{TinyMLClass::kStationary, 0.0f, false};
    if (!initialized_) {
        ESP_LOGE(TAG, "classify() called before init()");
        return result;
    }

    auto *interpreter = static_cast<tflite::MicroInterpreter *>(interpreter_);
    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);

    if ((size_t)input->bytes / sizeof(int8_t) != n_features) {
        ESP_LOGE(TAG, "Feature size mismatch: model expects %d, got %u",
                 (int)(input->bytes / sizeof(int8_t)), (unsigned)n_features);
        return result;
    }

    // Quantize float features -> int8 using the model's input quant params.
    const float in_scale = input->params.scale;
    const int32_t in_zp = input->params.zero_point;
    int8_t *in_data = input->data.int8;
    for (size_t i = 0; i < n_features; ++i) {
        int32_t q = (int32_t)lrintf(features[i] / in_scale) + in_zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        in_data[i] = (int8_t)q;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return result;
    }

    // Dequantize output, argmax.
    const float out_scale = output->params.scale;
    const int32_t out_zp = output->params.zero_point;
    const int8_t *out_data = output->data.int8;
    const int n_classes = output->bytes / sizeof(int8_t);

    int best_idx = 0;
    float best_prob = -1.0f;
    float probs[8];  // n_classes is 3; 8 is a safe ceiling
    float sum = 0.0f;
    for (int i = 0; i < n_classes && i < 8; ++i) {
        float p = (out_data[i] - out_zp) * out_scale;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        probs[i] = p;
        sum += p;
        if (p > best_prob) {
            best_prob = p;
            best_idx = i;
        }
    }
    // Renormalize (dequant rounding can leave the softmax not summing to 1).
    if (sum > 1e-6f) {
        best_prob /= sum;
    }

    result.cls = static_cast<TinyMLClass>(best_idx);
    result.confidence = best_prob;
    result.valid = true;
    return result;
}

const char *tinyml_class_to_string(TinyMLClass cls) {
    switch (cls) {
        case TinyMLClass::kStationary: return "stationary";
        case TinyMLClass::kNonStationary: return "non_stationary";
        case TinyMLClass::kSpeech: return "speech";
        default: return "unknown";
    }
}
