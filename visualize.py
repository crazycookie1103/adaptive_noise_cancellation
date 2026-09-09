import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft
from src.data_prep import build_scenario
from src.pipeline import run_pipeline

def generate_spectrograms():
    print("=======================================================================")
    print("           GENERATING ANC SPECTROGRAM EVIDENCE FIGURE                  ")
    print("=======================================================================\n")

    speech_files = glob.glob("data/clean_speech/**/*.flac", recursive=True) + \
                   glob.glob("data/clean_speech/**/*.wav", recursive=True)
    if not speech_files:
        print("[ERROR] No clean speech files found.")
        return

    speech_path = speech_files[0]
    print(f"Target Speech File: {speech_path}")

    noise_specs = [
        ("data/noise/ev_inverter_hum.wav", 0.2),
        ("data/noise/ev_road_tire_noise.wav", 1.0),
        ("data/noise/ev_pothole_thumps.wav", 0.3)
    ]

    # Build 0 dB scenario using current files
    speech, noise, fs = build_scenario(speech_path, noise_specs, snr_db=0.0)
    primary = speech + noise
    reference = noise

    print("Running pipeline for visual comparison...")
    enhanced = run_pipeline(primary, reference, fs)

    os.makedirs("outputs", exist_ok=True)
    sf.write("outputs/enhanced_visual_test.wav", enhanced, fs)
    print("Saved audio file: outputs/enhanced_visual_test.wav")

    # Compute STFT
    f, t, Zxx_clean = stft(speech, fs, nperseg=512, noverlap=256)
    _, _, Zxx_noisy = stft(primary, fs, nperseg=512, noverlap=256)
    _, _, Zxx_enh = stft(enhanced, fs, nperseg=512, noverlap=256)

    db_clean = 20 * np.log10(np.abs(Zxx_clean) + 1e-6)
    db_noisy = 20 * np.log10(np.abs(Zxx_noisy) + 1e-6)
    db_enh = 20 * np.log10(np.abs(Zxx_enh) + 1e-6)

    # Plot 3-panel comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    vmin, vmax = -80, 10

    axes[0].pcolormesh(t, f, db_noisy, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[0].set_title("1. Noisy Input (Primary Mic @ 0 dB SNR)")
    axes[0].set_ylabel("Freq (Hz)")

    axes[1].pcolormesh(t, f, db_enh, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[1].set_title("2. Enhanced Output (Full Active Pipeline)")
    axes[1].set_ylabel("Freq (Hz)")

    axes[2].pcolormesh(t, f, db_clean, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[2].set_title("3. Ground Truth Clean Speech")
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_xlabel("Time (seconds)")

    fig.tight_layout()
    out_path = "outputs/anc_spectrogram_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Graphical evidence saved successfully to: {out_path}")
    plt.show()

if __name__ == "__main__":
    generate_spectrograms()