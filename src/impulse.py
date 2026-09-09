import numpy as np
from scipy.stats import kurtosis

def detect_and_repair(x, fs, win_ms=5, kurt_thresh=8.0):
    win = int(fs * win_ms / 1000)
    out = x.copy()
    mask = np.zeros(len(x), dtype=bool)
    
    if win <= 0 or len(x) < win:
        return out, mask

    for i in range(0, len(x) - win, win):
        seg = x[i : i + win]
        
        # Calculate Fisher kurtosis (normal distribution = 0.0)
        if kurtosis(seg, fisher=True) > kurt_thresh:
            mask[i : i + win] = True
            
            # Boundary values for smooth linear interpolation
            left = x[i - 1] if i > 0 else x[0]
            right_idx = min(len(x) - 1, i + win)
            right = x[right_idx]
            
            # Interpolate smoothly across the damaged frame
            out[i : i + win] = np.linspace(left, right, win)
            
    return out, mask