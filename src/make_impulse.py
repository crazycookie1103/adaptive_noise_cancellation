import os
import numpy as np
import soundfile as sf

os.makedirs("data/noise", exist_ok=True)

fs = 16000
duration_sec = 5
n_samples = fs * duration_sec
rng = np.random.default_rng(42)

sig = np.zeros(n_samples, dtype=np.float32)

# Generate 4 road expansion joint "thumps" (damped 80Hz resonant wave)
t_impulse = np.linspace(0, 0.1, int(fs * 0.1))
low_frequency_thump = np.sin(2 * np.pi * 80 * t_impulse) * np.exp(-30 * t_impulse)

for idx in [20000, 42000, 58000, 71000]:
    sig[idx : idx + len(low_frequency_thump)] += low_frequency_thump * rng.uniform(1.5, 2.5)

sf.write("data/noise/ev_pothole_thumps.wav", sig, fs)
print("Generated EV pothole/expansion joint transients -> data/noise/ev_pothole_thumps.wav")