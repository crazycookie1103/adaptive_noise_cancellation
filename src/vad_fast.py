import numpy as np

def fast_energy_vad(x, fs, frame_ms=10, zcr_thresh=0.25):
    frame_len = int(fs * frame_ms / 1000)
    n_frames = len(x) // frame_len
    mask = np.zeros(len(x), dtype=bool)
    
    if n_frames == 0:
        return mask

    # 1. Compute frame energies to estimate noise floor adaptively
    frame_energies = []
    for i in range(n_frames):
        seg = x[i*frame_len:(i+1)*frame_len]
        energy_db = 10 * np.log10(np.mean(seg**2) + 1e-12)
        frame_energies.append(energy_db)

    # Threshold set 3 dB above median energy floor
    adaptive_thresh_db = np.median(frame_energies) + 3.0

    # 2. Evaluate speech presence per frame
    for i in range(n_frames):
        seg = x[i*frame_len:(i+1)*frame_len]
        energy_db = frame_energies[i]
        zcr = np.mean(np.abs(np.diff(np.sign(seg)))) / 2
        
        is_speech = (energy_db > adaptive_thresh_db) and (zcr < zcr_thresh)
        mask[i*frame_len:(i+1)*frame_len] = is_speech

    # Fill remainder samples to match full signal length
    if len(x) > n_frames * frame_len:
        mask[n_frames*frame_len:] = mask[(n_frames-1)*frame_len]

    return mask