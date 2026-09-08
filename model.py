"""
Model architecture: a tiny Depthwise-Separable CNN (DS-CNN), the same
family used in the TF speech-commands / keyword-spotting reference models
that are known to run comfortably on Cortex-M / ESP32-S3 class MCUs.

Input:  (28, 20, 1)  log-mel spectrogram (see features.py)
Output: 3-way softmax -> ["stationary", "non_stationary", "speech"]

Sizing target: after int8 post-training quantization, this should land
well under 100 KB — comfortable for ESP32-S3 (512 KB SRAM + PSRAM), with
plenty of headroom for the NLMS/Wiener DSP pipeline running alongside it.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = ["stationary", "non_stationary", "speech"]
INPUT_SHAPE = (28, 20, 1)


def build_model(input_shape=INPUT_SHAPE, n_classes=len(CLASSES), width=8):
    """
    width: base number of filters. 8 keeps the model very small; bump to
    12-16 if you find accuracy is the bottleneck rather than footprint.
    """
    inputs = layers.Input(shape=input_shape, name="log_mel_input")

    # Initial standard conv to mix across mel bins
    x = layers.Conv2D(width, kernel_size=(3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable blocks
    for filters, stride in [(width * 2, (2, 2)), (width * 4, (2, 2))]:
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride,
                                    padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size=(1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = models.Model(inputs, outputs, name="tinyml_anc_classifier")
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
    n_params = m.count_params()
    print(f"\nTotal params: {n_params:,} "
          f"(~{n_params / 1024:.1f} KB at float32, "
          f"~{n_params / 1024 / 4:.1f} KB at int8)")
