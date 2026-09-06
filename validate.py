import os
import glob
import csv
import numpy as np
import soundfile as sf

from pystoi import stoi

from src.impulse import detect_and_repair
from src.vad_fast import fast_energy_vad
from src.nlms import run_nlms
from src.spectral_mask import apply_spectral_mask


DATA_DIR = "data/mixed"
OUTPUT_DIR = "outputs"
RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)


SCENARIOS = [
    "ev_s1_highway_cruise",
    "ev_s2_rough_road_thumps",
    "ev_s3_full_cabin_mix",
    "solo_inverter_hum",
    "solo_road_tire_noise",
    "solo_pothole_thumps",
    "solo_gunshots_fireworks",
    "stress_s1_helicopter",
    "stress_s3_extreme_impulsive",
    "pair_helicopter_potholes",
    "pair_road_fireworks",
]


SNR_OPTIONS = {
    "neg3dB": -3.0,
    "0dB": 0.0,
    "3dB": 3.0,
}


CONFIGS = [
    ("Full Pipeline", True, True),
    ("No Stage 0", False, True),
    ("No Stage 3", True, False),
]


def calculate_snr(clean, signal):
    noise = signal - clean

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    return 10 * np.log10(
        clean_power / noise_power
    )


def run_custom_pipeline(
    primary,
    reference,
    fs,
    use_stage0=True,
    use_stage3=True
):

    if use_stage0:

        p_clean, _ = detect_and_repair(
            primary,
            fs,
            win_ms=5,
            kurt_thresh=8.0
        )

        r_clean, _ = detect_and_repair(
            reference,
            fs,
            win_ms=5,
            kurt_thresh=8.0
        )

    else:

        p_clean = primary.copy()
        r_clean = reference.copy()

    speech_mask = fast_energy_vad(
        p_clean,
        fs,
        frame_ms=10,
        zcr_thresh=0.25
    )

    stage1_out = run_nlms(
        p_clean,
        r_clean,
        adapt_mask=speech_mask,
        num_taps=128,
        mu=0.03
    )

    if use_stage3:

        final_out = apply_spectral_mask(
            stage1_out,
            fs,
            floor_gain=0.45
        )

    else:

        final_out = stage1_out

    return final_out


def get_speech_ids():

    pattern = os.path.join(
        DATA_DIR,
        "*_ev_s1_highway_cruise_snr0dB_clean.wav"
    )

    files = glob.glob(pattern)

    speech_ids = []

    for path in files:

        filename = os.path.basename(path)

        speech_id = filename.replace(
            "_ev_s1_highway_cruise_snr0dB_clean.wav",
            ""
        )

        speech_ids.append(speech_id)

    return sorted(speech_ids)


def find_dataset(
    speech_id,
    scenario,
    snr_tag
):

    prefix = os.path.join(
        DATA_DIR,
        f"{speech_id}_{scenario}_snr{snr_tag}"
    )

    clean_file = prefix + "_clean.wav"
    primary_file = prefix + "_primary.wav"
    reference_file = prefix + "_reference.wav"

    if not all(
        os.path.exists(path)
        for path in [
            clean_file,
            primary_file,
            reference_file
        ]
    ):
        return None

    return (
        clean_file,
        primary_file,
        reference_file
    )


def run_validation_suite():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 100)
    print(
        "                 EV CABIN ANC — FULL DATASET VALIDATION"
    )
    print("=" * 100)

    speech_ids = get_speech_ids()

    if not speech_ids:

        print(
            "\n[ERROR] No generated datasets found."
        )

        return

    print(
        f"\nSpeech recordings found: {len(speech_ids)}"
    )

    print(
        f"Scenarios: {len(SCENARIOS)}"
    )

    print(
        f"SNR levels: {len(SNR_OPTIONS)}"
    )

    print(
        f"Pipeline configurations: {len(CONFIGS)}"
    )

    results = []

    run_number = 0

    total_expected = (
        len(speech_ids)
        * len(SCENARIOS)
        * len(SNR_OPTIONS)
        * len(CONFIGS)
    )

    print(
        f"Maximum evaluation runs: {total_expected}"
    )

    print("\nStarting validation...\n")

    for speech_id in speech_ids:

        print("\n" + "=" * 100)
        print(
            f"SPEECH RECORDING: {speech_id}"
        )
        print("=" * 100)

        for scenario in SCENARIOS:

            print(
                f"\nScenario: {scenario}"
            )

            for snr_tag, snr_value in SNR_OPTIONS.items():

                dataset = find_dataset(
                    speech_id,
                    scenario,
                    snr_tag
                )

                if dataset is None:

                    print(
                        f"  [SKIP] Missing dataset "
                        f"for {snr_value:+.0f} dB"
                    )

                    continue

                clean_file, primary_file, reference_file = dataset

                clean, fs_clean = sf.read(
                    clean_file
                )

                primary, fs_primary = sf.read(
                    primary_file
                )

                reference, fs_reference = sf.read(
                    reference_file
                )

                if not (
                    fs_clean
                    == fs_primary
                    == fs_reference
                ):

                    print(
                        "  [ERROR] Sampling rates do not match."
                    )

                    continue

                fs = fs_clean

                n = min(
                    len(clean),
                    len(primary),
                    len(reference)
                )

                clean = clean[:n]
                primary = primary[:n]
                reference = reference[:n]

                # --------------------------------------------------
                # BASELINE — calculated BEFORE running ANC
                # --------------------------------------------------

                stoi_before = float(
                    stoi(
                        clean,
                        primary,
                        fs,
                        extended=False
                    )
                )

                snr_before = calculate_snr(
                    clean,
                    primary
                )

                for config_name, use_s0, use_s3 in CONFIGS:

                    run_number += 1

                    print(
                        f"  [{run_number}/{total_expected}] "
                        f"{snr_value:+.0f} dB | "
                        f"{config_name}",
                        end=" ... "
                    )

                    try:

                        enhanced = run_custom_pipeline(
                            primary,
                            reference,
                            fs,
                            use_stage0=use_s0,
                            use_stage3=use_s3
                        )

                        enhanced = np.asarray(
                            enhanced
                        ).squeeze()

                        n_out = min(
                            len(clean),
                            len(enhanced)
                        )

                        clean_eval = clean[:n_out]
                        enhanced_eval = enhanced[:n_out]

                        stoi_after = float(
                            stoi(
                                clean_eval,
                                enhanced_eval,
                                fs,
                                extended=False
                            )
                        )

                        snr_after = calculate_snr(
                            clean_eval,
                            enhanced_eval
                        )

                        stoi_delta = (
                            stoi_after
                            - stoi_before
                        )

                        snr_gain = (
                            snr_after
                            - snr_before
                        )

                        print(
                            f"STOI {stoi_before:.4f} → "
                            f"{stoi_after:.4f} | "
                            f"Δ {stoi_delta:+.4f} | "
                            f"SNR gain {snr_gain:+.2f} dB"
                        )

                        results.append({

                            "speech_id": speech_id,

                            "scenario": scenario,

                            "snr_db": snr_value,

                            "configuration": config_name,

                            "stoi_before": stoi_before,

                            "stoi_after": stoi_after,

                            "stoi_delta": stoi_delta,

                            "snr_before_db": snr_before,

                            "snr_after_db": snr_after,

                            "snr_gain_db": snr_gain,

                        })

                    except Exception as e:

                        print(
                            f"[ERROR] {e}"
                        )

    if not results:

        print(
            "\n[WARNING] No successful evaluations."
        )

        return

    fieldnames = [
        "speech_id",
        "scenario",
        "snr_db",
        "configuration",
        "stoi_before",
        "stoi_after",
        "stoi_delta",
        "snr_before_db",
        "snr_after_db",
        "snr_gain_db",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print("\n" + "=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)

    print(
        f"\nSuccessful evaluations: "
        f"{len(results)}"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    full_pipeline = [
        r for r in results
        if r["configuration"]
        == "Full Pipeline"
    ]

    if full_pipeline:

        print(
            "\nFULL PIPELINE OVERALL AVERAGE"
        )

        print("-" * 50)

        print(
            f"Average STOI before : "
            f"{np.mean([r['stoi_before'] for r in full_pipeline]):.4f}"
        )

        print(
            f"Average STOI after  : "
            f"{np.mean([r['stoi_after'] for r in full_pipeline]):.4f}"
        )

        print(
            f"Average STOI gain   : "
            f"{np.mean([r['stoi_delta'] for r in full_pipeline]):+.4f}"
        )

        print(
            f"Average SNR gain    : "
            f"{np.mean([r['snr_gain_db'] for r in full_pipeline]):+.2f} dB"
        )


if __name__ == "__main__":
    run_validation_suite()