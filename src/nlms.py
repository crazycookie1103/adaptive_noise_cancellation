import numpy as np

class VSSNLMS:
    def __init__(self, num_taps=128, mu_max=0.05, mu_min=0.001, eps=1e-4):
        self.num_taps = num_taps
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.eps = eps
        self.w = np.zeros(num_taps)
        self.ref_hist = np.zeros(num_taps)

    def step(self, ref_sample, primary_sample, adapt=True):
        self.ref_hist = np.roll(self.ref_hist, 1)
        self.ref_hist[0] = ref_sample
        
        y_hat = np.dot(self.w, self.ref_hist)
        error = primary_sample - y_hat
        
        # Only adapt filter weights when speech is absent (adapt=True)
        if adapt:
            norm = np.dot(self.ref_hist, self.ref_hist) + self.eps
            self.w += (self.mu_max / norm) * error * self.ref_hist
            
        return error

def run_nlms(primary, reference, adapt_mask=None, num_taps=128, mu=0.05):
    f = VSSNLMS(num_taps=num_taps, mu_max=mu)
    out = np.zeros_like(primary)
    
    for i in range(len(primary)):
        # If adapt_mask indicates speech (True/1), set adapt to False
        should_adapt = not bool(adapt_mask[i]) if adapt_mask is not None else True
        out[i] = f.step(reference[i], primary[i], adapt=should_adapt)
        
    return out