import numpy as np
from scipy.signal import stft, istft

def apply_spectral_mask(signal, fs, floor_gain=0.12, over_subtraction=1.8):
    f, t, Zxx = stft(signal, fs, nperseg=512, noverlap=256)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Calculate total energy per frame across all frequencies
    frame_energy = np.sum(mag**2, axis=0)
    
    # Identify the quietest 15% of time frames to isolate noise spectrum
    quiet_idx = np.argsort(frame_energy)[:max(1, int(len(frame_energy) * 0.15))]
    noise_est = np.mean(mag[:, quiet_idx], axis=1, keepdims=True)

    # Perform Power Spectral Subtraction
    signal_power = mag**2
    noise_power = noise_est**2
    
    subtracted_power = signal_power - (over_subtraction * noise_power)
    # Apply floor gain threshold
    subtracted_power = np.maximum(subtracted_power, (floor_gain**2) * signal_power)

    # Calculate Wiener Gain
    gain = np.sqrt(subtracted_power / (signal_power + 1e-10))
    gain = np.clip(gain, floor_gain, 1.0)

    # Reconstruct Output
    Zxx_enhanced = gain * mag * np.exp(1j * phase)
    _, enhanced_time = istft(Zxx_enhanced, fs, nperseg=512, noverlap=256)
    
    # Dimension matching
    if len(enhanced_time) > len(signal):
        enhanced_time = enhanced_time[:len(signal)]
    elif len(enhanced_time) < len(signal):
        enhanced_time = np.pad(enhanced_time, (0, len(signal) - len(enhanced_time)))

    return enhanced_time