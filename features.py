"""
Feature extraction for the TinyML noise/speech classifier.

Design constraints (ESP32-S3 target):
  - Sample rate: 16 kHz (mono)
  - Classification hop: 100 ms  -> matches control_interface.py's 100 ms poll
  - Analysis window: 300 ms of audio context per inference (three 100 ms
    hops of history are buffered on-device; this gives the model enough
    context to distinguish stationary hum from non-stationary rumble)
  - Frame length: 25 ms, frame step: 10 ms  -> ~28 frames per 300 ms window
  - 20 mel bins -> feature map is (28, 20, 1), ~560 int8 values in, tiny.

This is implemented with plain numpy (rfft + mel filterbank) rather than
librosa so the exact same math can be re-derived in fixed-point C on the
ESP32-S3 (librosa's implementation details are harder to port faithfully).
"""

import numpy as np

SAMPLE_RATE = 16000
WINDOW_MS = 300          # total audio context per inference
HOP_MS = 100             # matches TinyML control loop cadence
FRAME_LEN_MS = 25
FRAME_STEP_MS = 10
N_MELS = 20
FMIN_HZ = 50
FMAX_HZ = 7500           # < Nyquist (8000) for 16 kHz

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)      # 4800
FRAME_LEN = int(SAMPLE_RATE * FRAME_LEN_MS / 1000)        # 400
FRAME_STEP = int(SAMPLE_RATE * FRAME_STEP_MS / 1000)      # 160
N_FFT = 512                                               # next pow2 >= FRAME_LEN


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE,
                          fmin=FMIN_HZ, fmax=FMAX_HZ):
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


_MEL_FB = build_mel_filterbank()
_HANN = np.hanning(FRAME_LEN).astype(np.float32)


def log_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """
    audio: 1D float32 array, expected length == WINDOW_SAMPLES (will be
           zero-padded / truncated if not).
    returns: (n_frames, N_MELS) float32 array of log-mel energies.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[:WINDOW_SAMPLES]

    n_frames = 1 + (len(audio) - FRAME_LEN) // FRAME_STEP
    feats = np.zeros((n_frames, N_MELS), dtype=np.float32)

    for i in range(n_frames):
        start = i * FRAME_STEP
        frame = audio[start:start + FRAME_LEN] * _HANN
        spec = np.fft.rfft(frame, n=N_FFT)
        power = (spec.real ** 2 + spec.imag ** 2) / N_FFT
        mel_energy = _MEL_FB @ power
        feats[i] = np.log(mel_energy + 1e-6)

    return feats


def extract_features(audio, sr=SAMPLE_RATE):
    """Returns model-ready features, shape (n_frames, N_MELS, 1)."""
    feats = log_mel_spectrogram(audio, sr)
    # Per-utterance mean normalization (cheap, no dataset stats needed on-device)
    feats = feats - feats.mean()
    feats = feats / (feats.std() + 1e-6)
    return feats[..., np.newaxis].astype(np.float32)


if __name__ == "__main__":
    dummy = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 0.1
    f = extract_features(dummy)
    print("Feature shape:", f.shape)  # -> (28, 20, 1)
