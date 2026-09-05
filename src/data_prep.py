import numpy as np
import soundfile as sf

def load_and_match(path, target_len):
    sig, _ = sf.read(path)
    if sig.ndim > 1:
        sig = sig[:, 0]
    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)), mode='wrap')
    else:
        sig = sig[:target_len]
    return sig

def build_scenario(clean_path, noise_specs, snr_db=5.0):
    speech, fs = sf.read(clean_path)
    if speech.ndim > 1:
        speech = speech[:, 0]
    
    target_len = len(speech)
    mixed_noise = np.zeros(target_len)

    for noise_path, weight in noise_specs:
        noise_sig = load_and_match(noise_path, target_len)
        mixed_noise += weight * noise_sig

    # Standardize RMS energy scaling for exact target SNR
    speech_rms = np.sqrt(np.mean(speech**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(mixed_noise**2) + 1e-12)

    # Calculate required noise RMS for desired target SNR (dB)
    # SNR = 20 * log10(speech_rms / noise_rms)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    
    # Apply scale factor to noise
    mixed_noise = mixed_noise * (target_noise_rms / noise_rms)

    return speech, mixed_noise, fs