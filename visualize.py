import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft

from src.pipeline import run_pipeline


# ============================================================
# AVAILABLE DATASET SCENARIOS
# ============================================================

SCENARIOS = {
    "1": ("ev_s1_highway_cruise", "EV Highway Cruise"),
    "2": ("ev_s2_rough_road_thumps", "EV Rough Road Thumps"),
    "3": ("ev_s3_full_cabin_mix", "EV Full Cabin Mix"),
    "4": ("stress_s1_helicopter", "Helicopter"),
    "5": ("stress_s3_extreme_impulsive", "Extreme Impulsive"),
    "6": ("solo_gunshots_fireworks", "Gunshots + Fireworks"),
    "7": ("pair_helicopter_potholes", "Helicopter + Potholes"),
}


SNR_OPTIONS = {
    "1": ("neg3dB", "-3 dB"),
    "2": ("0dB", "0 dB"),
    "3": ("3dB", "+3 dB"),
}


# ============================================================
# DISPLAY SCENARIO MENU
# ============================================================

def choose_scenario():

    print("\n" + "=" * 65)
    print("                 ANC DATASET VISUALIZER")
    print("=" * 65)

    print("\nChoose a noise scenario:\n")

    for number, (_, description) in SCENARIOS.items():
        print(f"  {number}. {description}")

    while True:

        choice = input("\nEnter scenario number: ").strip()

        if choice in SCENARIOS:
            scenario, description = SCENARIOS[choice]

            print(f"\nSelected: {description}")

            return scenario, description

        print("[ERROR] Invalid choice. Please enter a number from 1 to 7.")


# ============================================================
# DISPLAY SNR MENU
# ============================================================

def choose_snr():

    print("\n" + "-" * 65)
    print("Choose SNR:\n")

    for number, (_, description) in SNR_OPTIONS.items():
        print(f"  {number}. {description}")

    while True:

        choice = input("\nEnter SNR number: ").strip()

        if choice in SNR_OPTIONS:
            snr, description = SNR_OPTIONS[choice]

            print(f"\nSelected SNR: {description}")

            return snr, description

        print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_dataset(scenario, snr):

    pattern = (
        f"data/mixed/*_{scenario}_snr{snr}_clean.wav"
    )

    matches = glob.glob(pattern)

    if not matches:

        print("\n[ERROR] Dataset not found.")
        print(f"Searched for:")
        print(pattern)

        return None

    clean_path = matches[0]

    primary_path = clean_path.replace(
        "_clean.wav",
        "_primary.wav"
    )

    reference_path = clean_path.replace(
        "_clean.wav",
        "_reference.wav"
    )

    if not os.path.exists(primary_path):
        print("[ERROR] Primary file missing.")
        return None

    if not os.path.exists(reference_path):
        print("[ERROR] Reference file missing.")
        return None

    return clean_path, primary_path, reference_path


# ============================================================
# VISUALIZE DATASET
# ============================================================

def visualize_dataset(
    scenario,
    scenario_description,
    snr,
    snr_description
):

    dataset = find_dataset(scenario, snr)

    if dataset is None:
        return

    clean_path, primary_path, reference_path = dataset

    print("\n" + "=" * 65)
    print("                    DATASET INFORMATION")
    print("=" * 65)

    print(f"\nScenario : {scenario_description}")
    print(f"SNR      : {snr_description}")

    print("\nFiles being used:")

    print(f"\nClean:")
    print(f"  {clean_path}")

    print(f"\nPrimary:")
    print(f"  {primary_path}")

    print(f"\nReference:")
    print(f"  {reference_path}")

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    clean, fs = sf.read(clean_path)
    primary, _ = sf.read(primary_path)
    reference, _ = sf.read(reference_path)

    print("\n" + "-" * 65)
    print("AUDIO INFORMATION")
    print("-" * 65)

    print(f"Sampling rate : {fs} Hz")
    print(f"Duration      : {len(clean) / fs:.2f} seconds")
    print(f"Samples       : {len(clean)}")

    # ========================================================
    # RUN ANC
    # ========================================================

    print("\nRunning ANC pipeline...")

    enhanced = run_pipeline(
        primary,
        reference,
        fs
    )

    print("ANC processing complete.")

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs("outputs", exist_ok=True)

    # ========================================================
    # SAVE ENHANCED AUDIO
    # ========================================================

    safe_scenario = scenario.replace(" ", "_")

    audio_path = (
        f"outputs/{safe_scenario}_{snr}_enhanced.wav"
    )

    sf.write(
        audio_path,
        enhanced,
        fs
    )

    print(f"\nEnhanced audio saved:")
    print(f"  {audio_path}")

    # ========================================================
    # TIME AXIS
    # ========================================================

    time = np.arange(len(clean)) / fs

    # ========================================================
    # WAVEFORM FIGURE
    # ========================================================

    print("\nGenerating waveform figure...")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True
    )

    axes[0].plot(time, primary)

    axes[0].set_title(
        f"Noisy Input — {scenario_description} — {snr_description}"
    )

    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time, enhanced)

    axes[1].set_title(
        "Enhanced Output — ANC Pipeline"
    )

    axes[1].set_ylabel("Amplitude")

    axes[2].plot(time, clean)

    axes[2].set_title(
        "Ground Truth — Clean Speech"
    )

    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (seconds)")

    fig.suptitle(
        "Time-Domain Waveform Comparison",
        fontsize=14
    )

    fig.tight_layout()

    waveform_path = (
        f"outputs/{safe_scenario}_{snr}_waveform.png"
    )

    plt.savefig(
        waveform_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Waveform saved:")
    print(f"  {waveform_path}")

    plt.show()

    # ========================================================
    # STFT
    # ========================================================

    print("\nGenerating spectrograms...")

    f, t, Z_clean = stft(
        clean,
        fs,
        nperseg=512,
        noverlap=256
    )

    _, _, Z_primary = stft(
        primary,
        fs,
        nperseg=512,
        noverlap=256
    )

    _, _, Z_enhanced = stft(
        enhanced,
        fs,
        nperseg=512,
        noverlap=256
    )

    # ========================================================
    # CONVERT TO dB
    # ========================================================

    db_clean = 20 * np.log10(
        np.abs(Z_clean) + 1e-6
    )

    db_primary = 20 * np.log10(
        np.abs(Z_primary) + 1e-6
    )

    db_enhanced = 20 * np.log10(
        np.abs(Z_enhanced) + 1e-6
    )

    # ========================================================
    # SPECTROGRAM FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )

    vmin = -80
    vmax = 10

    axes[0].pcolormesh(
        t,
        f,
        db_primary,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno"
    )

    axes[0].set_title(
        f"1. Noisy Input — {scenario_description} — {snr_description}"
    )

    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].pcolormesh(
        t,
        f,
        db_enhanced,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno"
    )

    axes[1].set_title(
        "2. Enhanced Output — ANC Pipeline"
    )

    axes[1].set_ylabel("Frequency (Hz)")

    axes[2].pcolormesh(
        t,
        f,
        db_clean,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno"
    )

    axes[2].set_title(
        "3. Ground Truth — Clean Speech"
    )

    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (seconds)")

    fig.suptitle(
        "ANC Spectrogram Comparison",
        fontsize=14
    )

    fig.tight_layout()

    spectrogram_path = (
        f"outputs/{safe_scenario}_{snr}_spectrogram.png"
    )

    plt.savefig(
        spectrogram_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Spectrogram saved:")
    print(f"  {spectrogram_path}")

    plt.show()

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 65)
    print("                  VISUALIZATION COMPLETE")
    print("=" * 65)

    print("\nYou visualized:")
    print(f"  Scenario : {scenario_description}")
    print(f"  SNR      : {snr_description}")

    print("\nGenerated:")
    print("  1. Enhanced audio")
    print("  2. Waveform comparison")
    print("  3. Spectrogram comparison")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    scenario, scenario_description = choose_scenario()

    snr, snr_description = choose_snr()

    visualize_dataset(
        scenario,
        scenario_description,
        snr,
        snr_description
    )