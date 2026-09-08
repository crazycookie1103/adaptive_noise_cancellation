import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from features import extract_features, SAMPLE_RATE, WINDOW_SAMPLES
from model import CLASSES
from train import predict_with_thresholds

def run_wav_inference(wav_path, model_path="tinyml_anc.keras"):
    # 1. Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # 2. Read audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE} Hz, but got {sr} Hz")
    
    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 3. Slide 300 ms windows with 100 ms hop size (1600 samples)
    hop_samples = int(SAMPLE_RATE * 0.100) # 100 ms stride
    predictions = []

    print(f"\nProcessing '{wav_path}' ({len(audio)/sr:.2f} seconds)...")
    print("-" * 55)
    print(f"{'Time (s)':<10} | {'Stationary':<11} | {'Non-Stat':<10} | {'Speech':<8} | {'Prediction'}")
    print("-" * 55)

    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, hop_samples):
        chunk = audio[start : start + WINDOW_SAMPLES]
        
        # Extract features (28, 20, 1) and expand batch dimension to (1, 28, 20, 1)
        feat = extract_features(chunk)
        input_tensor = np.expand_dims(feat, axis=0)

        # Predict raw probabilities
        probs = model.predict(input_tensor, verbose=0)
        
        # Custom threshold decision
        pred_idx = predict_with_thresholds(probs, speech_threshold=0.75, non_stationary_threshold=0.60)[0]
        predicted_label = CLASSES[pred_idx]

        timestamp = start / SAMPLE_RATE
        p_stat, p_nonstat, p_speech = probs[0]

        print(f"{timestamp:<10.2f} | {p_stat:<11.2f} | {p_nonstat:<10.2f} | {p_speech:<8.2f} | {predicted_label}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_wav.py <path_to_audio.wav>")
    else:
        run_wav_inference(sys.argv[1])
