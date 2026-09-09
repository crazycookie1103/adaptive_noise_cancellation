import numpy as np
import pyaudio
import time
from src.pipeline import Pipeline  # Or import individual stages directly

class RealtimeANCEngine:
    def __init__(self, sample_rate=16000, block_size=512):
        self.sr = sample_rate
        self.block_size = block_size
        
        # Initialize your existing pipeline/DSP objects
        # Adjust parameters according to your src/control_interface.py
        self.pipeline = Pipeline(sample_rate=self.sr) 
        
        # Audio Stream Configuration
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio stream callback processing audio block-by-block.
        """
        # 1. Convert byte stream from input/mic to float numpy array
        audio_frame = np.frombuffer(in_data, dtype=np.float32)
        
        # 2. Extract primary and reference channels (assuming stereo mic input)
        # If single input, synthesize or map accordingly
        primary = audio_frame[0::2]   # Desired + Noise (e.g. cabin mic)
        reference = audio_frame[1::2] # Noise reference (e.g. engine/accel sensor)
        
        # 3. Process through your repository DSP stages
        # Step 0: Impulse repair
        # Step 1: NLMS
        # Step 2: Spectral Masking
        cleaned_frame, anti_noise_frame = self.pipeline.process_frame(primary, reference)
        
        # 4. Convert output back to PCM bytes for audio speaker playback
        out_bytes = cleaned_frame.astype(np.float32).tobytes()
        
        return (out_bytes, pyaudio.paContinue)

    def start_stream(self):
        self.is_running = True
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,             # Stereo: [Primary, Reference]
            rate=self.sr,
            input=True,
            output=True,
            frames_per_buffer=self.block_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.is_running = False