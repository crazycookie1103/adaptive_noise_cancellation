import os
import glob
import soundfile as sf

from src.data_prep import build_scenario
from src.room_sim import simulate_two_mic_room

os.makedirs("data/mixed", exist_ok=True)

scenarios = {
    # Existing EV Cabin Scenarios
    "ev_s1_highway_cruise": [
        ("data/noise/ev_inverter_hum.wav", 0.7),
        ("data/noise/ev_road_tire_noise.wav", 0.5),
    ],

    "ev_s2_rough_road_thumps": [
        ("data/noise/ev_inverter_hum.wav", 0.5),
        ("data/noise/ev_pothole_thumps.wav", 1.0),
    ],

    "ev_s3_full_cabin_mix": [
        ("data/noise/ev_inverter_hum.wav", 0.6),
        ("data/noise/ev_road_tire_noise.wav", 0.6),
        ("data/noise/ev_pothole_thumps.wav", 0.8),
    ],

    # Stress Test & Extreme Transient Scenarios
    "stress_s1_helicopter": [
        ("data/noise/helicopter.wav", 1.0),
    ],

    "stress_s2_gunshot_transients": [
        ("data/noise/ev_road_tire_noise.wav", 0.4),
        ("data/noise/gunshot.wav", 1.0),
    ],

    "stress_s3_extreme_impulsive": [
        ("data/noise/helicopter.wav", 0.5),
        ("data/noise/gunshots_fireworks.flac", 0.8),
    ],

    # Individual Noise Profiles
    "solo_inverter_hum": [
        ("data/noise/ev_inverter_hum.wav", 1.0),
    ],

    "solo_road_tire_noise": [
        ("data/noise/ev_road_tire_noise.wav", 1.0),
    ],

    "solo_pothole_thumps": [
        ("data/noise/ev_pothole_thumps.wav", 1.0),
    ],

    "solo_gunshot": [
        ("data/noise/gunshot.wav", 1.0),
    ],

    "solo_gunshots_fireworks": [
        ("data/noise/gunshots_fireworks.flac", 1.0),
    ],

    # Extra Combined Noise Pairs
    "pair_helicopter_potholes": [
        ("data/noise/helicopter.wav", 0.7),
        ("data/noise/ev_pothole_thumps.wav", 0.8),
    ],

    "pair_inverter_gunshot": [
        ("data/noise/ev_inverter_hum.wav", 0.6),
        ("data/noise/gunshot.wav", 0.9),
    ],

    "pair_road_fireworks": [
        ("data/noise/ev_road_tire_noise.wav", 0.5),
        ("data/noise/gunshots_fireworks.flac", 0.8),
    ],
}


# Find all clean speech files
speech_files = (
    glob.glob("data/clean_speech/**/*.flac", recursive=True)
    + glob.glob("data/clean_speech/**/*.wav", recursive=True)
)

if not speech_files:
    print("[ERROR] No speech files found under data/clean_speech/")
else:
    print(f"Found {len(speech_files)} clean speech file(s).")
    print(f"Found {len(scenarios)} scenario definitions.")

    for speech_path in speech_files:
        spk_id = os.path.splitext(os.path.basename(speech_path))[0]

        for name, noise_specs in scenarios.items():

            # Verify all required noise files exist
            missing_files = [
                path for path, _ in noise_specs
                if not os.path.exists(path)
            ]

            if missing_files:
                print(
                    f"[SKIP] Scenario '{name}' missing noise files: "
                    f"{missing_files}"
                )
                continue

            for snr in [-3.0, 0.0, 3.0]:

                if snr < 0:
                    snr_tag = f"neg{abs(int(snr))}"
                else:
                    snr_tag = f"{int(snr)}"

                speech, noise, fs = build_scenario(
                    speech_path,
                    noise_specs,
                    snr_db=snr
                )

                primary, reference, clean_primary = simulate_two_mic_room(
                    speech,
                    noise,
                    fs,
                    rt60=0.15,
                    target_snr_db=snr
                )

                out_prefix = (
                    f"data/mixed/"
                    f"{spk_id}_{name}_snr{snr_tag}dB"
                )

                sf.write(
                   f"{out_prefix}_clean.wav", clean_primary, fs
                )
               

                sf.write(
                    f"{out_prefix}_primary.wav",
                    primary,
                    fs
                )

                sf.write(
                    f"{out_prefix}_reference.wav",
                    reference,
                    fs
                )

                print(f"Generated sample set: {out_prefix}")