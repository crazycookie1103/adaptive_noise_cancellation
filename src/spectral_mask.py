import numpy as np
from scipy.signal import stft, istft

def apply_spectral_mask(signal, fs, floor_gain=0.18, over_subtraction=1.3):
    f, t, Zxx = stft(signal, fs, nperseg=512, noverlap=256)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise floor from the lowest 15% energy frames
    noise_est = np.mean(np.sort(mag, axis=1)[:, :max(1, int(mag.shape[1] * 0.15))], axis=1, keepdims=True)

    # Balanced over-subtraction
    subtracted_mag = mag**2 - (over_subtraction * (noise_est**2))
    subtracted_mag = np.maximum(subtracted_mag, (floor_gain**2) * (mag**2))

    # Wiener Gain Filter
    gain = np.sqrt(subtracted_mag / (mag**2 + 1e-10))
    gain = np.clip(gain, floor_gain, 1.0)

    # Reconstruct Signal
    Zxx_enhanced = gain * mag * np.exp(1j * phase)
    _, enhanced_time = istft(Zxx_enhanced, fs, nperseg=512, noverlap=256)
    
    # Match output length
    if len(enhanced_time) > len(signal):
        enhanced_time = enhanced_time[:len(signal)]
    elif len(enhanced_time) < len(signal):
        enhanced_time = np.pad(enhanced_time, (0, len(signal) - len(enhanced_time)))

    return enhanced_time