import os
import soundfile as sf
from src.data_prep import build_scenario
from src.pipeline import run_pipeline
from src.evaluate import evaluate

def main():
    os.makedirs("outputs", exist_ok=True)

    clean_speech_path = "data/clean_speech/1272-128104-0000.flac"
    
    # Noise specifications with strong broadband road noise to test masking
    noise_specs = [
        ("data/noise/ev_inverter_hum.wav", 0.2),
        ("data/noise/ev_road_tire_noise.wav", 1.0),  # Heavy broadband tire rumble
        ("data/noise/ev_pothole_thumps.wav", 0.3)
    ]

    print("Building heavy broadband noise scenario (Target SNR: 0.0 dB)...")
    
    # 1. Generate clean speech and combined noise track at 0 dB SNR
    speech, noise, fs = build_scenario(
        clean_path=clean_speech_path,
        noise_specs=noise_specs,
        snr_db=0.0
    )

    # 2. Define microphone signals
    primary_mic = speech + noise
    reference_mic = noise

    print("Running multi-stage ANC pipeline...")
    # 3. Process through DSP pipeline
    enhanced = run_pipeline(primary_mic, reference_mic, fs)

    # 4. Save both noisy input and enhanced output for listening/comparison
    sf.write("outputs/noisy_primary.wav", primary_mic, fs)
    sf.write("outputs/enhanced.wav", enhanced, fs)

    # 5. Evaluate benchmark metrics
    metrics = evaluate(
        clean=speech,
        noisy=primary_mic,
        enhanced=enhanced,
        fs=fs
    )

    print("\n--- Benchmark Metrics (0 dB Heavy Broadband Scenario) ---")
    print(metrics)
    print("\nAudio files written to outputs/:")
    print("  - outputs/noisy_primary.wav (Listen to input before ANC)")
    print("  - outputs/enhanced.wav      (Listen to output after ANC)")

if __name__ == "__main__":
    main()