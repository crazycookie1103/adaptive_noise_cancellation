import os
import numpy as np
import soundfile as sf

# Ensure target directories exist
os.makedirs("data/clean_speech", exist_ok=True)
os.makedirs("data/noise", exist_ok=True)

fs = 16000
duration = 10.0  # Seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1. Synthesize EV Inverter Hum (120 Hz fundamental + harmonics)
print("Synthesizing EV Inverter Hum proxy...")
inverter = (
    0.6 * np.sin(2 * np.pi * 120 * t) +
    0.3 * np.sin(2 * np.pi * 240 * t) +
    0.15 * np.sin(2 * np.pi * 360 * t) +
    0.1 * np.random.normal(0, 1, len(t))
)
sf.write("data/noise/ev_inverter_hum.wav", inverter.astype(np.float32), fs)
print("Generated: data/noise/ev_inverter_hum.wav")

# 2. Synthesize EV Road/Tire Noise (Low-pass filtered pavement roll)
print("Synthesizing EV Road/Tire Noise proxy...")
raw_noise = np.random.normal(0, 1, len(t))
b = 0.05
tire_noise = np.zeros_like(raw_noise)
for i in range(1, len(raw_noise)):
    tire_noise[i] = b * raw_noise[i] + (1 - b) * tire_noise[i-1]

sf.write("data/noise/ev_road_tire_noise.wav", tire_noise.astype(np.float32), fs)
print("Generated: data/noise/ev_road_tire_noise.wav")

# 3. Synthesize EV Pothole Thumps (Impulsive low-frequency transients)
print("Synthesizing EV Pothole Thumps proxy...")
pothole = np.zeros_like(t)
impact_times = [1.5, 4.2, 7.8]  # Transient impact timestamps in seconds
for t_impact in impact_times:
    idx = int(t_impact * fs)
    decay_len = int(0.15 * fs)  # 150 ms duration
    if idx + decay_len < len(t):
        t_decay = np.linspace(0, 0.15, decay_len, endpoint=False)
        envelope = np.exp(-35 * t_decay)
        transient = np.sin(2 * np.pi * 45 * t_decay) * envelope
        pothole[idx:idx + decay_len] += transient * 1.5

sf.write("data/noise/ev_pothole_thumps.wav", pothole.astype(np.float32), fs)
print("Generated: data/noise/ev_pothole_thumps.wav")

print("\nAll EV noise assets generated successfully in data/noise/!")