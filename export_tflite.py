"""
Export a trained Keras model to a fully int8-quantized .tflite file
suitable for TFLite Micro on the ESP32-S3, and then to a C header
(xxd-style byte array) for compiling directly into firmware.

Usage:
    python3 export_tflite.py tinyml_anc.keras data/
    -> writes tinyml_anc_int8.tflite and tinyml_anc_model.h, calibrated
       on real recordings from data/ (falls back to synthetic calibration
       with a warning if data_dir is omitted)
"""

import sys

import numpy as np
import tensorflow as tf

from features import extract_features, WINDOW_SAMPLES
from train import make_synthetic_dataset, load_wav_dataset


def representative_dataset_gen(n_samples=100, data_dir=None):
    """
    Calibration data for full-integer quantization. This determines the
    int8 scale/zero-point for every layer -- if it doesn't reflect the
    real signal statistics your model will actually see, quantization
    error goes up even though the float model was fine. Always prefer
    `data_dir` (real recordings) over the synthetic fallback once you
    have any real data at all, even a small amount.
    """
    if data_dir:
        X, _, _ = load_wav_dataset(data_dir)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))[:n_samples]
        X = X[idx]
    else:
        print("  [warn] no data_dir given to representative_dataset_gen -- "
              "calibrating on SYNTHETIC data. Int8 accuracy on real audio "
              "may be noticeably worse than the float model until you pass "
              "a real data_dir here.")
        X, _, _ = make_synthetic_dataset(n_per_class=n_samples // 3 + 1)

    for i in range(min(n_samples, len(X))):
        yield [X[i:i + 1].astype(np.float32)]


def export(keras_path, tflite_out="tinyml_anc_int8.tflite", data_dir=None):
    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(data_dir=data_dir)
    # Force full int8 (weights + activations + I/O) -- required for
    # TFLite Micro / ESP-DL, which does not support float fallback ops.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_out, "wb") as f:
        f.write(tflite_model)

    print(f"Wrote {tflite_out} ({len(tflite_model):,} bytes)")
    return tflite_out


def to_c_array(tflite_path, header_out="tinyml_anc_model.h", var_name="g_tinyml_anc_model"):
    with open(tflite_path, "rb") as f:
        data = f.read()

    lines = [
        "// Auto-generated from {} -- do not edit by hand.".format(tflite_path),
        "#ifndef TINYML_ANC_MODEL_H_",
        "#define TINYML_ANC_MODEL_H_",
        "",
        "alignas(8) const unsigned char {}[] = {{".format(var_name),
    ]
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};")
    lines.append("")
    lines.append("#endif  // TINYML_ANC_MODEL_H_")

    with open(header_out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {header_out} ({len(data):,} bytes as C array)")


def sanity_check(tflite_path):
    """Runs one dummy inference through the exported int8 model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    dummy_audio = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 0.1
    feats = extract_features(dummy_audio)  # float32, (28, 20, 1)

    scale, zero_point = in_detail["dtype"], None
    in_scale, in_zp = in_detail["quantization"]
    quantized = np.round(feats / in_scale + in_zp).astype(np.int8)

    interpreter.set_tensor(in_detail["index"], quantized[np.newaxis, ...])
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])[0]

    out_scale, out_zp = out_detail["quantization"]
    probs = (out.astype(np.float32) - out_zp) * out_scale

    print("Sanity-check output shape:", out.shape, "dtype:", out_detail["dtype"])
    print("Dequantized class probabilities:", np.round(probs, 3))


if __name__ == "__main__":
    keras_path = sys.argv[1] if len(sys.argv) > 1 else "tinyml_anc.keras"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else None
    tflite_path = export(keras_path, data_dir=data_dir)
    to_c_array(tflite_path)
    sanity_check(tflite_path)
