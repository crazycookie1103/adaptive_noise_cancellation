import sys
import numpy as np
import tensorflow as tf
import soundfile as sf

from features import extract_features, SAMPLE_RATE, WINDOW_SAMPLES
from model import CLASSES
from train import predict_with_thresholds


# =====================================================================
# DSP STAGES (SIMULATED FOR LAPTOP TESTING)
# =====================================================================

def stage0_kurtosis_clamp(chunk, threshold=3.0):
    """Stage 0: Simple transient spike clamp (e.g., door slams)."""
    std = np.std(chunk) + 1e-8
    mean = np.mean(chunk)
    # Clip extreme spikes exceeding standard deviation multiplier
    return np.clip(chunk, mean - threshold * std, mean + threshold * std)


def stage1_vss_nlms_filter(chunk, step_size=0.01, filter_order=32):
    """Stage 1: Sample-by-sample VSS-NLMS adaptive filter."""
    # Simple synthetic noise estimator for desktop evaluation
    w = np.zeros(filter_order)
    x_buf = np.zeros(filter_order)
    clean_chunk = np.zeros_like(chunk)
    
    # Simple noise reference (low-pass smoothed copy)
    ref_noise = np.convolve(chunk, np.ones(5)/5.0, mode='same')

    for n in range(len(chunk)):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = ref_noise[n]
        
        y = np.dot(w, x_buf)               # Estimated noise
        e = chunk[n] - y                   # Error signal (Desired - Estimated)
        norm = np.dot(x_buf, x_buf) + 1e-6 # Power normalization
        w += (step_size / norm) * e * x_buf # Tap update
        clean_chunk[n] = e

    return clean_chunk


def stage2_wiener_spectral_mask(chunk, oversubtraction=1.0, is_speech=False):
    """Stage 2: Frequency-domain Wiener post-filter."""
    if is_speech:
        return chunk # Preserve speech formants without aggressive masking

    # Convert to frequency domain
    spectrum = np.fft.rfft(chunk)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Estimate noise floor magnitude
    noise_est = np.mean(magnitude) * 0.3
    
    # Calculate Wiener gain mask with over-subtraction factor alpha
    subtracted = magnitude**2 - (oversubtraction * (noise_est**2))
    subtracted = np.maximum(subtracted, 0.01 * (magnitude**2)) # Floor mask
    
    gain = np.sqrt(subtracted) / (magnitude + 1e-8)
    gain = np.clip(gain, 0.05, 1.0) # Prevent zeroing out completely

    # Reconstruct clean audio signal
    clean_spectrum = gain * magnitude * np.exp(1j * phase)
    return np.fft.irfft(clean_spectrum, n=len(chunk))


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def process_audio_file(input_wav, output_wav="clean_output.wav", model_path="tinyml_anc.keras"):
    print(f"\nLoading model '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    audio, sr = sf.read(input_wav, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    hop_samples = int(SAMPLE_RATE * 0.100) # 100 ms hop size
    processed_audio = np.zeros_like(audio)
    
    print(f"Processing '{input_wav}' ({len(audio)/sr:.2f} s)...")
    print("-" * 65)
    print(f"{'Time (s)':<8} | {'Prediction':<15} | {'NLMS mu':<8} | {'Wiener alpha':<12}")
    print("-" * 65)

    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, hop_samples):
        chunk = audio[start : start + WINDOW_SAMPLES]

        # 1. Extract TinyML features and predict
        feat = extract_features(chunk)
        probs = model.predict(np.expand_dims(feat, axis=0), verbose=0)
        
        pred_idx = predict_with_thresholds(
            probs, speech_threshold=0.75, non_stationary_threshold=0.60
        )[0]
        predicted_label = CLASSES[pred_idx]

        # 2. Dynamic parameter allocation based on predictions
        if predicted_label == "speech":
            mu = 0.001
            alpha = 0.5
            is_speech = True
        elif predicted_label == "non_stationary":
            mu = 0.08
            alpha = 2.2
            is_speech = False
        else: # stationary
            mu = 0.01
            alpha = 1.0
            is_speech = False

        # 3. Pass through 3-Stage DSP Pipeline
        s0_out = stage0_kurtosis_clamp(chunk)
        s1_out = stage1_vss_nlms_filter(s0_out, step_size=mu)
        s2_out = stage2_wiener_spectral_mask(s1_out, oversubtraction=alpha, is_speech=is_speech)

        # Write window back to audio buffer (Overlap-Add)
        processed_audio[start : start + WINDOW_SAMPLES] = s2_out

        timestamp = start / SAMPLE_RATE
        print(f"{timestamp:<8.2f} | {predicted_label:<15} | {mu:<8.3f} | {alpha:<12.1f}")

    # Normalize and write output WAV file
    processed_audio = processed_audio / (np.max(np.abs(processed_audio)) + 1e-8)
    sf.write(output_wav, processed_audio, SAMPLE_RATE)
    
    print("-" * 65)
    print(f"DONE! Cleaned audio saved to: '{output_wav}'\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_anc_dsp.py <path_to_noisy_audio.wav>")
    else:
        process_audio_file(sys.argv[1])
