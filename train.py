"""
Key Upgrades:
- Fixed speech class allocation during train/val splitting via chunk grouping
- Added audio data augmentation (gain scaling + stationary noise injection)
- Scaled class weights with a safety cap to prevent gradient explosion
- Added custom confidence thresholding during post-evaluation
- Re-added: guard against a class having too few source files to split
  (previously caused "0 train / N val" -> crash once val had every window)
"""

import glob
import os

import numpy as np
import tensorflow as tf

from features import extract_features, SAMPLE_RATE, WINDOW_SAMPLES
from model import build_model, CLASSES

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False


def augment_audio(chunk, rng, class_name):
    """Applies fast, on-the-fly numpy data augmentation."""
    # Random volume/gain variation (0.7x to 1.3x)
    gain = rng.uniform(0.7, 1.3)
    chunk = chunk * gain

    # Inject minor Gaussian noise to stationary audio to aid generalization
    if class_name == "stationary" and rng.random() > 0.5:
        noise_level = rng.uniform(0.001, 0.005)
        chunk += noise_level * rng.standard_normal(len(chunk))

    return np.clip(chunk, -1.0, 1.0).astype(np.float32)


def load_wav_dataset(data_dir, augment=True, seed=42):
    """
    Loads real recordings from data_dir/<class_name>/*.wav.
    Extracts overlapping windows and assigns source group IDs for group-based splitting.
    """
    if not HAVE_SOUNDFILE:
        raise RuntimeError("pip install soundfile --break-system-packages")

    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    group_id = 0

    for label_idx, class_name in enumerate(CLASSES):
        files = glob.glob(os.path.join(data_dir, class_name, "*.wav"))
        if not files:
            print(f"  [warn] no files found for class '{class_name}' in {data_dir}")

        for path in files:
            audio, sr = sf.read(path, dtype="float32")
            if sr != SAMPLE_RATE:
                raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")

            # 50% overlap sliding window
            step = WINDOW_SAMPLES // 2
            n = max(1, 1 + (len(audio) - WINDOW_SAMPLES) // step)

            for i in range(n):
                start = i * step
                chunk = audio[start:start + WINDOW_SAMPLES]

                # Ensure fixed frame length
                if len(chunk) < WINDOW_SAMPLES:
                    chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

                if augment:
                    chunk = augment_audio(chunk, rng, class_name)

                X.append(extract_features(chunk))
                y.append(label_idx)
                groups.append(group_id)

            group_id += 1  # Unique ID per file to prevent data leakage

    if not X:
        raise RuntimeError(
            f"No .wav files found anywhere under '{data_dir}' for any class "
            f"(stationary/non_stationary/speech). Check the path is correct "
            f"and relative (e.g. 'data/', not '/data') and that each class "
            f"subfolder actually contains .wav files."
        )

    return np.stack(X), np.array(y, dtype=np.int64), np.array(groups, dtype=np.int64)


def make_synthetic_dataset(n_per_class=500, seed=0):
    """Crude synthetic dataset for pipeline smoke-testing."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE

    for label_idx, class_name in enumerate(CLASSES):
        for _ in range(n_per_class):
            if class_name == "stationary":
                freq = rng.uniform(80, 200)
                audio = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(WINDOW_SAMPLES)
            elif class_name == "non_stationary":
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)
                audio = mod * 0.3 * rng.standard_normal(WINDOW_SAMPLES)
            else:  # speech
                f0 = rng.uniform(90, 250)
                audio = sum(
                    (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
                    for k in range(1, 6)
                )
                envelope = (np.sin(2 * np.pi * rng.uniform(3, 8) * t) > 0).astype(np.float32)
                audio = 0.3 * audio * envelope + 0.02 * rng.standard_normal(WINDOW_SAMPLES)

            audio = audio.astype(np.float32)
            X.append(extract_features(audio))
            y.append(label_idx)

    X, y = np.stack(X), np.array(y, dtype=np.int64)
    groups = np.arange(len(y), dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], groups[perm]


def split_by_group(y, groups, val_frac=0.15, seed=0):
    """
    Stratified group split ensuring intact source files per partition.

    FIX: the previous version took max(1, round(val_frac * n_groups))
    groups for val with no upper bound. When a class has only 1 source
    file, that's max(1, 0) = 1 -- the entire file -- leaving ZERO for
    train. This caps n_val_groups so at least 1 group always remains for
    train, and falls back to an all-train, no-val split (with a warning)
    when a class has fewer than 2 source files.
    """
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)

    for class_idx in np.unique(y):
        class_rows = np.where(y == class_idx)[0]
        class_groups = np.unique(groups[class_rows])
        rng.shuffle(class_groups)

        if len(class_groups) < 2:
            print(f"  [warn] class {int(class_idx)} ({CLASSES[int(class_idx)]}) has only "
                  f"{len(class_groups)} source file(s) -- all of it goes to "
                  f"train, no val split possible for this class yet. Add a "
                  f"2nd+ file per class for a real validation read.")
            n_val_groups = 0
        else:
            n_val_groups = max(1, int(round(val_frac * len(class_groups))))
            n_val_groups = min(n_val_groups, len(class_groups) - 1)  # keep >=1 for train

        val_groups = set(class_groups[:n_val_groups])

        for row in class_rows:
            if groups[row] in val_groups:
                val_mask[row] = True
            else:
                train_mask[row] = True

    return train_mask, val_mask


def compute_bounded_class_weights(y_train, max_weight=12.0):
    """Computes inverse class weights capped at a safe ceiling."""
    counts = np.bincount(y_train, minlength=len(CLASSES))
    total = len(y_train)
    weights = {}

    for i, count in enumerate(counts):
        if count > 0:
            w = total / (len(CLASSES) * count)
            
            '''weights[i] = min(w, max_weight)  # Cap weight scale'''
            
            if CLASSES[i] == "stationary":
                w *= 1.25
            weights[i] = min(w, max_weight)
        else:
            weights[i] = 1.0

    return weights


def predict_with_thresholds(probabilities, speech_threshold=0.75, non_stationary_threshold=0.60):
    """Custom confidence thresholding to suppress non-stationary false positives."""
    preds = []
    for prob in probabilities:
        # Check high-priority speech detector target
        if prob[CLASSES.index("speech")] >= speech_threshold:
            preds.append(CLASSES.index("speech"))
        # Require higher confidence for non_stationary predictions
        elif prob[CLASSES.index("non_stationary")] >= non_stationary_threshold:
            preds.append(CLASSES.index("non_stationary"))
        else:
            preds.append(CLASSES.index("stationary"))
    return np.array(preds)


def train(X, y, groups=None, epochs=35, batch_size=64,
          out_path="tinyml_anc.keras", use_class_weight=True):
    if groups is None:
        groups = np.arange(len(y), dtype=np.int64)

    train_mask, val_mask = split_by_group(y, groups)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Split: {len(y_train)} train / {len(y_val)} val windows "
          f"({len(np.unique(groups[train_mask]))} / "
          f"{len(np.unique(groups[val_mask]))} source files)")
    print("Per-class counts -- train:", np.bincount(y_train, minlength=len(CLASSES)),
          " val:", np.bincount(y_val, minlength=len(CLASSES)))

    if len(y_train) == 0:
        raise RuntimeError(
            "No training data after the split. Add more .wav files -- at "
            "least 2-3 per class is the practical minimum -- and try again."
        )
    if len(y_val) == 0:
        print("  [warn] no validation data at all -- every class had only "
              "1 source file. Training will proceed but you have no way "
              "to judge real accuracy until you add more files.")

    class_weight = None
    if use_class_weight:
        class_weight = compute_bounded_class_weights(y_train)
        print("Capped class weights:", class_weight)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    has_val = len(y_val) > 0
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if has_val else "loss",
            patience=3, factor=0.5, min_lr=1e-5),
    ]
    if has_val:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True))

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if has_val else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    model.save(out_path)
    print(f"\nSaved trained model to {out_path}")
    return model, (X_val, y_val)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    

    if data_dir:
        print(f"Loading real dataset from {data_dir} ...")
        X, y, groups = load_wav_dataset(data_dir, augment=True)
    else:
        print("No data_dir given -> generating SYNTHETIC smoke-test dataset.\n")
        X, y, groups = make_synthetic_dataset()

    print(f"Dataset: {X.shape[0]} samples, feature shape {X.shape[1:]}")
    model, (X_val, y_val) = train(X, y, groups)

    if len(y_val) == 0:
        print("\nNo validation set was available (see warning above) -- "
              "skipping accuracy/confusion-matrix reporting.")
    else:
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nStandard Validation accuracy: {val_acc:.3f}")

        try:
            from sklearn.metrics import classification_report, confusion_matrix
            probs = model.predict(X_val, verbose=0)

            # Argmax baseline evaluation
            preds_argmax = probs.argmax(axis=1)
            print("\n--- Standard Argmax Evaluation ---")
            print(classification_report(y_val, preds_argmax, target_names=CLASSES, zero_division=0))

            # Custom threshold evaluation
            preds_thresh = predict_with_thresholds(probs)
            print("\n--- Custom Threshold Evaluation ---")
            print(classification_report(y_val, preds_thresh, target_names=CLASSES, zero_division=0))
            print("Confusion matrix (rows=true, cols=predicted):")
            print(CLASSES)
            print(confusion_matrix(y_val, preds_thresh))

        except ImportError:
            print("\n(Install scikit-learn for metric analysis: pip install scikit-learn --break-system-packages)")
            
            
            
  
