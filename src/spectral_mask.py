import numpy as np
import scipy.signal as signal

def apply_spectral_mask(audio, fs, floor_gain=0.45):
    """
    Wiener post-filter with relaxed floor gain to protect speech formants.
    """
    f, t, Zxx = signal.stft(audio, fs=fs, nperseg=512, noverlap=384)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise power floor from low-energy frames
    noise_pow = np.percentile(magnitude ** 2, 10, axis=1, keepdims=True)
    signal_pow = magnitude ** 2

    # Wiener Gain Mask with Floor Protection
    gain = signal_pow / (signal_pow + noise_pow + 1e-12)
    gain = np.maximum(gain, floor_gain)

    # Reconstruct audio
    Zxx_clean = gain * magnitude * np.exp(1j * phase)
    _, enhanced = signal.istft(Zxx_clean, fs=fs, nperseg=512, noverlap=384)

    n = min(len(audio), len(enhanced))
    out = np.zeros_like(audio)
    out[:n] = enhanced[:n]
    return out