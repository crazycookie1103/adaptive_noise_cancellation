// components/tinyml_control/control_interface.cpp

#include "control_interface.h"

ControlState::ControlState(float step_smooth)
    : step_smooth_(step_smooth),
      smoothed_{/*mu=*/0.20f, /*floor_gain=*/0.35f} {
    // Start in the non_stationary (moderate) profile until the first
    // confident classification arrives -- a deliberately conservative
    // default rather than starting fully aggressive or fully frozen.
}

SuppressionParams ControlState::target_for(TinyMLClass cls) {
    switch (cls) {
        case TinyMLClass::kStationary:
            return {0.50f, 0.15f};
        case TinyMLClass::kNonStationary:
            return {0.20f, 0.35f};
        case TinyMLClass::kSpeech:
            return {0.05f, 0.60f};
        default:
            return {0.20f, 0.35f};
    }
}

void ControlState::update(const TinyMLResult &result) {
    if (!result.valid || result.confidence < kConfidenceThreshold) {
        // Hold last smoothed params -- matches control_interface.py's
        // "Messages with confidence < 0.60 are ignored" rule.
        return;
    }

    SuppressionParams target = target_for(result.cls);

    // EMA: smoothed = smoothed + step_smooth * (target - smoothed)
    smoothed_.mu += step_smooth_ * (target.mu - smoothed_.mu);
    smoothed_.floor_gain += step_smooth_ * (target.floor_gain - smoothed_.floor_gain);
}
