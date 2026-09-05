import numpy as np
from src.impulse import detect_and_repair
from src.vad_fast import fast_energy_vad
from src.nlms import run_nlms
from src.spectral_mask import apply_spectral_mask

def run_pipeline(primary, reference, fs):
    # Stage 0: Transient Repair (Kurtosis-based)
    p_clean, _ = detect_and_repair(primary, fs, win_ms=5, kurt_thresh=8.0)
    r_clean, _ = detect_and_repair(reference, fs, win_ms=5, kurt_thresh=8.0)

    # Compute binary VAD mask using fast_energy_vad
    speech_mask = fast_energy_vad(p_clean, fs, frame_ms=10, zcr_thresh=0.25)

    # Stage 1: NLMS Adaptive Filter (Freeze adaptation during active speech)
    stage1_out = run_nlms(p_clean, r_clean, adapt_mask=speech_mask, num_taps=128, mu=0.05)

    # Stage 3: Spectral Wiener Post-Filter (Preserve speech formants with floor_gain=0.45)
    final_out = apply_spectral_mask(stage1_out, fs, floor_gain=0.45)

    return final_out