import numpy as np

class ControlState:
    """
    Consumes environment classifications from CNN/TinyML model 
    (e.g., {'class': 'stationary', 'confidence': 0.85}) 
    and smoothly adapts DSP parameters to avoid audio artifacts.
    """
    def __init__(self, alpha=0.15):
        # Target values set by classifier
        self.target_mu = 0.3
        self.target_floor_gain = 0.25
        
        # Smoothed values actually used by DSP blocks
        self.mu = 0.3
        self.floor_gain = 0.25
        
        # Smoothing coefficient (lower = smoother parameter transitions)
        self.alpha = alpha

    def update(self, msg):
        """Processes incoming classification message from neural network."""
        if not msg or "class" not in msg or "confidence" not in msg:
            return

        cls = msg["class"]
        conf = msg["confidence"]

        # Confidence Threshold Check
        if conf < 0.6:
            return

        # Map acoustic environment class to target DSP parameters
        if cls == "stationary":
            self.target_mu = 0.5
            self.target_floor_gain = 0.15
        elif cls == "non_stationary":
            self.target_mu = 0.2
            self.target_floor_gain = 0.35
        elif cls == "speech":
            self.target_mu = 0.05
            self.target_floor_gain = 0.60

    def step_smooth(self):
        """
        Call this inside your audio frame processing loop.
        Applies Exponential Moving Average (EMA) to prevent audio clicks/pops.
        """
        self.mu += self.alpha * (self.target_mu - self.mu)
        self.floor_gain += self.alpha * (self.target_floor_gain - self.floor_gain)
        return self.mu, self.floor_gain