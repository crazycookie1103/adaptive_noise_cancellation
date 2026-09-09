import os
import numpy as np
import soundfile as sf

os.makedirs("data/clean_speech", exist_ok=True)
os.makedirs("data/noise", exist_ok=True)

# Check LibriSpeech clips (already downloaded in previous step)
speech_file = "data/clean_speech/1272-128104-0000.flac"
if os.path.exists(speech_file):
    print("LibriSpeech speech clips ready.")
else:
    print("Speech clips missing. Ensure 1272-128104-0000.flac is in data/clean_speech/")

fs = 16000
duration = 10  # 10 seconds of synthetic EV background audio
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1. Synthesize EV Inverter Hum (120 Hz fundamental + switching harmonics + low rumble)
print("Synthesizing EV Inverter Hum proxy...")
inverter = (
    0.6 * np.sin(2 * np.pi * 120 * t) +       # Primary motor hum
    0.3 * np.sin(2 * np.pi * 240 * t) +       # 2nd Harmonic
    0.15 * np.sin(2 * np.pi * 360 * t) +      # 3rd Harmonic
    0.1 * np.random.normal(0, 1, len(t))      # Ambient floor
)
sf.write("data/noise/ev_inverter_hum.wav", inverter.astype(np.float32), fs)
print("Generated: data/noise/ev_inverter_hum.wav")

# 2. Synthesize EV Road/Tire Noise (Low-pass filtered white noise for pavement roll)
print("Synthesizing EV Road/Tire Noise proxy...")
raw_noise = np.random.normal(0, 1, len(t))
# Exponential moving average filter to simulate low-frequency road resonance
b = 0.05
tire_noise = np.zeros_like(raw_noise)
for i in range(1, len(raw_noise)):
    tire_noise[i] = b * raw_noise[i] + (1 - b) * tire_noise[i-1]

sf.write("data/noise/ev_road_tire_noise.wav", tire_noise.astype(np.float32), fs)
print("Generated: data/noise/ev_road_tire_noise.wav")

print("\nAll EV dataset assets ready in data/!")