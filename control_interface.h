// components/tinyml_control/include/control_interface.h
//
// C++ mirror of src/control_interface.py's ControlState, so the ESP32-S3
// firmware and the Python reference implementation apply IDENTICAL
// smoothing/mapping logic. If you change thresholds or targets in
// control_interface.py, update the constants here too.

#pragma once

#include "tinyml_inference.h"

struct SuppressionParams {
    float mu;          // NLMS step size
    float floor_gain;  // Wiener post-filter floor gain
};

class ControlState {
public:
    // step_smooth: EMA smoothing coefficient in (0, 1]. Matches
    // control_interface.py's step_smooth -- keep these equal or the two
    // implementations will diverge in transition behavior.
    explicit ControlState(float step_smooth = 0.2f);

    // Feed one TinyML classification result (100-200 ms cadence).
    // Applies the >= 0.60 confidence gate internally: low-confidence
    // results are ignored and the previous smoothed params are held.
    void update(const TinyMLResult &result);

    // Current smoothed parameters to hand to the NLMS/Wiener DSP stages,
    // called from the real-time sample-by-sample audio loop.
    const SuppressionParams &current_params() const { return smoothed_; }

private:
    static constexpr float kConfidenceThreshold = 0.60f;

    // Target mapping per the interface contract:
    //   stationary:     mu=0.50, floor_gain=0.15  (aggressive -- inverter hum/HVAC)
    //   non_stationary:  mu=0.20, floor_gain=0.35  (moderate -- road/tire rumble)
    //   speech:          mu=0.05, floor_gain=0.60  (freeze NLMS, preserve formants)
    static SuppressionParams target_for(TinyMLClass cls);

    float step_smooth_;
    SuppressionParams smoothed_;
};
