import numpy as np
from src.impulse import detect_and_repair
from src.vad_fast import fast_energy_vad
from src.nlms import run_nlms
from src.spectral_mask import apply_spectral_mask

def run_pipeline(primary, reference, fs):
    # Stage 0: Transient Repair (Kurtosis-based)
    p_clean, _ = detect_and_repair(primary, fs, win_ms=5, kurt_thresh=8.0)
    r_clean, _ = detect_and_repair(reference, fs, win_ms=5, kurt_thresh=8.0)

    # Stage 1: Tight VAD mask (zcr_thresh=0.18 for sharper pause isolation)
    speech_mask = fast_energy_vad(p_clean, fs, frame_ms=10, zcr_thresh=0.18)

    # Stage 2: High-Resolution NLMS (512 taps for deep attenuation under -10 dB SNR)
    stage1_out = run_nlms(p_clean, r_clean, adapt_mask=speech_mask, num_taps=512, mu=0.12)

    # Stage 3: Frame-Energy VAD Spectral Masking
    final_out = apply_spectral_mask(stage1_out, fs, floor_gain=0.15, over_subtraction=1.5)

    return final_out