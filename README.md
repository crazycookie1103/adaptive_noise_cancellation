# TinyML Control Engine — Model Pipeline

Implements the async classifier feeding `src/control_interface.py`, matching
the interface contract exactly:

```json
{"class": "stationary", "confidence": 0.85}
```

Classes: `stationary`, `non_stationary`, `speech` — **no `impulsive` class**;
those transients are stripped upstream by Stage 0's kurtosis filter.

## Pipeline

| File | Purpose |
|---|---|
| `features.py` | 16 kHz audio → 20-bin log-mel spectrogram, 300 ms window / 100 ms hop → `(28, 20, 1)` |
| `model.py` | Tiny DS-CNN (~1.3 KB float32, 3-class softmax) |
| `train.py` | Training loop; loads real `.wav` data OR generates a synthetic smoke-test set |
| `export_tflite.py` | Full int8 post-training quantization → `.tflite` (10.4 KB) + C header for firmware |
| `infer_stub.py` | Python-side reference of the on-device inference loop; streams JSON matching the contract, one line per 100 ms hop |

Everything above has been run end-to-end on synthetic data as a pipeline
smoke test — shapes, training, quantization, and JSON output all verified.
**Accuracy numbers from synthetic data are meaningless**; they only prove
the plumbing works.

## Next steps (in order)

1. **Collect real audio.** You need labeled clips for the 3 classes:
   - `stationary`: HVAC, inverter hum, engine idle, steady road noise
   - `non_stationary`: tire rumble over varying terrain, wind gusts, chatter
   - `speech`: clean + noisy speech (VAD-positive segments)
   Keep potholes/door-slams/gunfire **out** — that's Stage 0's job.
   Aim for at least a few hundred clips per class, several seconds each,
   recorded (or at least dominated by) the same mic characteristics as
   the primary channel in deployment.

2. **Retrain on real data:**
   ```bash
   python3 train.py data/
   python3 export_tflite.py tinyml_anc.keras
   ```

3. **Re-check the confidence threshold.** The 0.60 threshold in the
   contract was presumably chosen for a reasonably confident model —
   validate it against your real validation set's confidence
   distribution once trained; tune per-class if one class is
   systematically under-confident.

4. **Port `infer_stub.py`'s loop to firmware.** The `.tflite` file and
   `tinyml_anc_model.h` C array are ready for TFLite Micro on ESP32-S3.
   You'll need to reimplement `features.py`'s log-mel extraction in
   fixed-point C for the device — the FFT/mel-filterbank math is written
   plainly in `features.py` specifically so it's easy to port faithfully
   (ESP-DL has FFT and mel-filterbank helpers you can use directly, or a
   CMSIS-DSP-style port).

5. **Validate confidence filtering + EMA smoothing end-to-end** by piping
   `infer_stub.py`'s stdout into `src/control_interface.py`:
   ```bash
   python3 infer_stub.py | python3 src/control_interface.py
   ```

## Firmware (ESP32-S3, ESP-IDF + TFLite Micro)

`firmware/` is an ESP-IDF component structure implementing the full
on-device loop: mic audio -> features -> inference -> smoothed
suppression params, ready to plug into your NLMS/Wiener DSP loop.

```
firmware/
  CMakeLists.txt
  components/tinyml_control/
    CMakeLists.txt
    include/
      tinyml_features.h       feature extraction (mirrors features.py exactly)
      tinyml_inference.h      TFLite Micro wrapper
      control_interface.h     C++ port of control_interface.py's ControlState
      tinyml_anc_constants.h  generated: Hann window + mel filterbank (bit-exact w/ training)
      tinyml_anc_model.h      generated: the int8 model as a C array
    tinyml_features.cpp
    tinyml_inference.cpp
    control_interface.cpp
  main/
    CMakeLists.txt
    idf_component.yml         pulls in espressif/esp-tflite-micro + espressif/esp-dsp
    tinyml_control_task.cpp   FreeRTOS task wiring everything together
```

**Important — this is NOT compile-verified.** I don't have an ESP-IDF
toolchain available to build and flash it, so treat this as a carefully
written first draft, not tested firmware. Specific things to check when
you build it:

- `esp-dsp`'s exact FFT API (`dsps_fft2r_fc32`, `dsps_bit_rev2r_fc32`,
  `dsps_fft2r_init_fc32`) — verify against whatever esp-dsp version
  `idf.py add-dependency` pulls in; function signatures have shifted
  across versions.
- `esp-tflite-micro`'s `MicroMutableOpResolver` op names
  (`AddConv2D`, `AddDepthwiseConv2D`, etc.) — confirm these match your
  installed version; op registration APIs change between TFLite Micro
  releases.
- `kTensorArenaSize` (32 KB) is a starting guess — the init log prints
  `arena_used_bytes()`, so shrink it to match once you've flashed it.
- **Cross-core synchronization**: `tinyml_get_current_params()` is a
  plain accessor with no locking. If your real-time audio loop runs on
  a different core than the TinyML task (likely, given the task is
  pinned to core 1), add a mutex or switch `SuppressionParams`' fields
  to `std::atomic<float>` before relying on it — a torn read across
  cores is a real bug, not a hypothetical one.
- `audio_ring_buffer_read()` in `tinyml_control_task.cpp` is a stub you
  need to wire to your actual mic capture / Stage-0 output buffer.

Feature extraction correctness matters more than firmware polish here:
`tinyml_anc_constants.h` is generated directly from `features.py`'s mel
filterbank and Hann window, so as long as `tinyml_features.cpp`'s FFT
and normalization steps are faithful to `features.py`'s (which I
followed closely), what the device sees should match training data
distribution. If accuracy on-device looks worse than in Python
evaluation, that mismatch is the first thing to check.

## Model capacity headroom

At `width=8` the model is ~1.3 KB (float32) / ~10 KB (int8, mostly
overhead from the flatbuffer format, not weights). This is far smaller
than the ESP32-S3 needs to worry about — if real-data accuracy is
insufficient, increase `width` in `model.py` (try 12 or 16) before
reaching for a fundamentally different architecture.
