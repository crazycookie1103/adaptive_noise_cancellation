import numpy as np
from pystoi import stoi

def evaluate(clean, noisy, enhanced, fs):
    # Ensure 1D arrays
    clean = np.squeeze(clean)
    noisy = np.squeeze(noisy)
    enhanced = np.squeeze(enhanced)

    # Align signal lengths
    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]

    # Calculate Signal-to-Noise Ratio (SNR) in dB
    def calculate_snr(ref, sig):
        noise = sig - ref
        power_ref = np.mean(ref ** 2) + 1e-12
        power_noise = np.mean(noise ** 2) + 1e-12
        return 10 * np.log10(power_ref / power_noise)

    return {
        "snr_before_db": round(float(calculate_snr(clean, noisy)), 2),
        "snr_after_db": round(float(calculate_snr(clean, enhanced)), 2),
        "stoi_before": round(float(stoi(clean, noisy, fs, extended=False)), 4),
        "stoi_after": round(float(stoi(clean, enhanced, fs, extended=False)), 4),
    }