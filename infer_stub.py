"""
Reference inference loop for the TinyML control engine.

This is the Python-side equivalent of what will eventually run as C/C++
on the ESP32-S3 (TFLite Micro interpreter, same .tflite file). Use this
to validate src/control_interface.py against realistic timing and output
statistics before porting to firmware.

Matches the interface contract exactly:
    {"class": "stationary", "confidence": 0.85}
    valid class strings: "stationary", "non_stationary", "speech"
    (never "impulsive" -- that's stripped upstream by Stage 0)

Run standalone to stream one JSON line every 100 ms to stdout, e.g.:
    python3 infer_stub.py | python3 src/control_interface.py
"""

import json
import sys
import time

import numpy as np
import tensorflow as tf

from features import extract_features, SAMPLE_RATE, WINDOW_SAMPLES, HOP_MS
from model import CLASSES


class TinyMLClassifier:
    def __init__(self, tflite_path="tinyml_anc_int8.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_detail = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.in_detail["quantization"]
        self.out_scale, self.out_zp = self.out_detail["quantization"]

        # Rolling audio buffer: holds the last WINDOW_SAMPLES of audio,
        # advanced by one HOP_MS (100 ms) worth of new samples each call.
        self._buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._hop_samples = int(SAMPLE_RATE * HOP_MS / 1000)

    def push_audio_and_classify(self, new_samples):
        """
        new_samples: float32 array of ~HOP_MS worth of new audio
                     (from the primary mic channel, post Stage-0 impulse filter).
        Returns dict matching the JSON contract.
        """
        new_samples = np.asarray(new_samples, dtype=np.float32)
        n = len(new_samples)
        self._buffer = np.roll(self._buffer, -n)
        self._buffer[-n:] = new_samples

        feats = extract_features(self._buffer)  # (28, 20, 1) float32
        quantized = np.round(feats / self.in_scale + self.in_zp).astype(np.int8)

        self.interpreter.set_tensor(self.in_detail["index"], quantized[np.newaxis, ...])
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_detail["index"])[0]
        probs = (out.astype(np.float32) - self.out_zp) * self.out_scale
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum()  # renormalize after dequant rounding

        top_idx = int(np.argmax(probs))
        return {
            "class": CLASSES[top_idx],
            "confidence": round(float(probs[top_idx]), 2),
        }


def stream_from_mic_or_file(classifier, audio_source, realtime=True):
    """
    audio_source: 1D float32 array of the full recording (16 kHz mono).
    Feeds it in HOP_MS chunks and prints one JSON line per hop, matching
    the on-device cadence.
    """
    hop = classifier._hop_samples
    n_hops = len(audio_source) // hop

    for i in range(n_hops):
        chunk = audio_source[i * hop:(i + 1) * hop]
        result = classifier.push_audio_and_classify(chunk)
        print(json.dumps(result), flush=True)
        if realtime:
            time.sleep(HOP_MS / 1000.0)


if __name__ == "__main__":
    # Demo: classify a short synthetic clip so this is runnable with no
    # external files. Swap this for a real 16kHz wav via soundfile.read().
    from train import make_synthetic_dataset

    clf = TinyMLClassifier("tinyml_anc_int8.tflite")

    print("Demo run on a few synthetic windows (not real audio):", file=sys.stderr)
    rng = np.random.default_rng(1)
    demo_audio = rng.standard_normal(WINDOW_SAMPLES * 5).astype(np.float32) * 0.1
    stream_from_mic_or_file(clf, demo_audio, realtime=False)
