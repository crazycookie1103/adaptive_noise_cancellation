import os
import glob
import numpy as np
import soundfile as sf
from src.data_prep import build_scenario
from src.pipeline import run_pipeline
from src.evaluate import evaluate
from src.impulse import detect_and_repair
from src.vad_fast import fast_energy_vad
from src.nlms import run_nlms
from src.spectral_mask import apply_spectral_mask

def run_custom_pipeline(primary, reference, fs, use_stage0=True, use_stage3=True):
    if use_stage0:
        p_clean, _ = detect_and_repair(primary, fs, win_ms=5, kurt_thresh=8.0)
        r_clean, _ = detect_and_repair(reference, fs, win_ms=5, kurt_thresh=8.0)
    else:
        p_clean, r_clean = primary.copy(), reference.copy()

    speech_mask = fast_energy_vad(p_clean, fs, frame_ms=10, zcr_thresh=0.25)
    stage1_out = run_nlms(p_clean, r_clean, adapt_mask=speech_mask, num_taps=128, mu=0.05)

    if use_stage3:
        final_out = apply_spectral_mask(stage1_out, fs, floor_gain=0.45)
    else:
        final_out = stage1_out

    return final_out

def run_validation_suite():
    print("===================================================================================")
    print("          EV CABIN ANC: TARGETED MULTI-SNR SPOT CHECK (-3dB, 0dB, +3dB)             ")
    print("===================================================================================\n")

    speech_files = glob.glob("data/clean_speech/**/*.flac", recursive=True) + \
                   glob.glob("data/clean_speech/**/*.wav", recursive=True)

    if not speech_files:
        print("[ERROR] No speech files found under data/clean_speech/")
        return

    # Cap to the first file to prevent extensive iteration loops
    speech_path = speech_files[0]
    filename = os.path.basename(speech_path)

    noise_specs = [
        ("data/noise/ev_inverter_hum.wav", 0.2),
        ("data/noise/ev_road_tire_noise.wav", 1.0),
        ("data/noise/ev_pothole_thumps.wav", 0.3)
    ]

    target_snrs = [-3.0, 0.0, 3.0]
    configs = [
        ("Full Pipeline", True, True),
        ("No Stage 0 (Transient Repair Off)", False, True),
        ("No Stage 3 (Wiener Floor Off)", True, False),
    ]

    header = f"{'Configuration':<35} | {'SNR':<6} | {'STOI In':<9} | {'STOI Out':<9} | {'Delta':<8} | {'SNR Gain':<9}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for config_name, use_s0, use_s3 in configs:
        for snr in target_snrs:
            speech, noise, fs = build_scenario(speech_path, noise_specs, snr_db=snr)
            primary = speech + noise
            reference = noise

            enhanced = run_custom_pipeline(primary, reference, fs, use_stage0=use_s0, use_stage3=use_s3)
            metrics = evaluate(clean=speech, noisy=primary, enhanced=enhanced, fs=fs)

            s_in = metrics['stoi_before']
            s_out = metrics['stoi_after']
            s_diff = s_out - s_in
            s_gain = metrics['snr_after_db'] - metrics['snr_before_db']

            print(f"{config_name:<35} | {snr:<+5.1f} | {s_in:<9.4f} | {s_out:<9.4f} | {f'+{s_diff:.4f}':<8} | {f'+{s_gain:.2f}dB':<9}")

    print("-" * len(header))
    print("\n[Validation Complete] Spot-check executed without long execution loops.\n")

if __name__ == "__main__":
    run_validation_suite()