import os
import soundfile as sf
from src.data_prep import build_scenario
from src.room_sim import simulate_two_mic_room

# Ensure target folder exists
os.makedirs("data/mixed", exist_ok=True)

# EV Acoustic Test Scenarios
scenarios = {
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
}

speech_path = "data/clean_speech/1272-128104-0000.flac"

for name, noise_paths in scenarios.items():
    # Mix speech and noise at 3dB SNR
    speech, noise, fs = build_scenario(speech_path, noise_paths, snr_db=3)
    
    # Simulate dual-microphone cabin acoustics
    primary, reference = simulate_two_mic_room(speech, noise, fs, rt60=0.15)

    # Save output audio files for benchmark testing
    sf.write(f"data/mixed/{name}_clean.wav", speech, fs)
    sf.write(f"data/mixed/{name}_primary.wav", primary, fs)
    sf.write(f"data/mixed/{name}_reference.wav", reference, fs)
    print(f"Successfully generated EV scenario: {name}")