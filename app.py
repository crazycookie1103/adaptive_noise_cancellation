import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft
import streamlit as st

# Direct imports from your project modules
from src.data_prep import build_scenario
from src.evaluate import evaluate
from src.pipeline import run_pipeline

st.set_page_config(layout="wide", page_title="ANC Pipeline Dashboard")

st.title("Adaptive Noise Cancellation (ANC) Dashboard")
st.markdown("Interactive UI connected directly to `src/pipeline.py` and `src/visualize.py`.")

# Sidebar Controls
st.sidebar.header("Scenario & Pipeline Parameters")

# 1. Input Clean Speech File Selection
speech_files = glob.glob("data/clean_speech/**/*.flac", recursive=True) + \
               glob.glob("data/clean_speech/**/*.wav", recursive=True)

if not speech_files:
    st.error("No clean speech audio files found in `data/clean_speech/`!")
    st.stop()

selected_speech = st.sidebar.selectbox("Select Clean Speech File", speech_files)

# 2. Target SNR Slider
snr_db = st.sidebar.slider("Target SNR (dB)", min_value=-10.0, max_value=15.0, value=0.0, step=1.0)

# 3. Noise Profile Weighting Sliders
st.sidebar.subheader("Noise Component Weights")
w_inverter = st.sidebar.slider("Inverter Hum Weight", 0.0, 2.0, 0.2, 0.1)
w_road = st.sidebar.slider("Road & Tire Noise Weight", 0.0, 2.0, 1.0, 0.1)
w_pothole = st.sidebar.slider("Pothole Thumps Weight", 0.0, 2.0, 0.3, 0.1)

noise_specs = [
    ("data/noise/ev_inverter_hum.wav", w_inverter),
    ("data/noise/ev_road_tire_noise.wav", w_road),
    ("data/noise/ev_pothole_thumps.wav", w_pothole)
]

def render_spectrogram_panel(speech, primary, enhanced, fs):
    f, t, Zxx_clean = stft(speech, fs, nperseg=512, noverlap=256)
    _, _, Zxx_noisy = stft(primary, fs, nperseg=512, noverlap=256)
    _, _, Zxx_enh = stft(enhanced, fs, nperseg=512, noverlap=256)

    db_clean = 20 * np.log10(np.abs(Zxx_clean) + 1e-6)
    db_noisy = 20 * np.log10(np.abs(Zxx_noisy) + 1e-6)
    db_enh = 20 * np.log10(np.abs(Zxx_enh) + 1e-6)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    vmin, vmax = -80, 10

    axes[0].pcolormesh(t, f, db_noisy, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[0].set_title("1. Noisy Input (Primary Mic)")
    axes[0].set_ylabel("Freq (Hz)")

    axes[1].pcolormesh(t, f, db_enh, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[1].set_title("2. Enhanced Output (Full Active Pipeline)")
    axes[1].set_ylabel("Freq (Hz)")

    axes[2].pcolormesh(t, f, db_clean, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[2].set_title("3. Ground Truth Clean Speech")
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_xlabel("Time (seconds)")

    fig.tight_layout()
    return fig

def analyze_anti_noise(primary, reference, enhanced):
    """
    Analyzes the synthesized anti-noise wave and cancellation efficiency.
    Returns calculated values derived directly from signal analysis.
    """
    # 1. Synthesized Anti-Noise Signal (Difference between noisy input and output)
    anti_noise = primary - enhanced

    # 2. Power / Energy attenuation calculation
    p_primary_noise = np.var(primary)
    p_residual = np.var(enhanced) + 1e-10
    attenuation_db = 10 * np.log10(p_primary_noise / p_residual)
    attenuation_percent = max(0.0, min(100.0, (1 - (p_residual / (p_primary_noise + 1e-10))) * 100))

    # 3. Cross-correlation between reference noise and anti-noise wave
    if np.std(reference) > 0 and np.std(anti_noise) > 0:
        correlation = float(np.corrcoef(reference, anti_noise)[0, 1])
    else:
        correlation = 0.0

    # 4. Phase Inversion Accuracy (Dot Product alignment)
    phase_accuracy = float(np.mean(np.sign(reference) != np.sign(anti_noise)) * 100)

    return {
        "cancellation_db": attenuation_db,
        "attenuation_percent": attenuation_percent,
        "correlation": correlation,
        "phase_accuracy": phase_accuracy
    }

# Run ANC Pipeline Button
if st.button("Run Pipeline & Update Dashboard", type="primary"):
    with st.spinner("Processing audio through DSP pipeline & analyzing anti-noise system..."):
        # Generate noisy primary & reference inputs via src/data_prep.py
        speech, noise, fs = build_scenario(selected_speech, noise_specs, snr_db=snr_db)
        primary = speech + noise
        reference = noise

        # Execute full DSP pipeline via src/pipeline.py
        enhanced = run_pipeline(primary, reference, fs)

        # Calculate metrics using src/evaluate.py
        metrics = evaluate(clean=speech, noisy=primary, enhanced=enhanced, fs=fs)
        snr_in = metrics.get('snr_before_db', 0.0)
        snr_out = metrics.get('snr_after_db', 0.0)
        snr_gain = snr_out - snr_in
        
        stoi_in = metrics.get('stoi_before', 0.0)
        stoi_out = metrics.get('stoi_after', 0.0)

        # Anti-Noise Deep Analysis
        anti_noise_stats = analyze_anti_noise(primary, reference, enhanced)

        # 1. Standard Performance Metrics
        st.subheader("1. Pipeline Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Input SNR", f"{snr_in:.2f} dB")
        col2.metric("Output SNR", f"{snr_out:.2f} dB")
        col3.metric("SNR Gain", f"+{snr_gain:.2f} dB")
        col4.metric("Input STOI", f"{stoi_in:.3f}")
        col5.metric("Enhanced STOI", f"{stoi_out:.3f}", delta=f"{stoi_out - stoi_in:+.3f}")

        # 2. Anti-Noise System Analysis Metrics
        st.subheader("2. Anti-Noise System Analysis")
        an_col1, an_col2, an_col3, an_col4 = st.columns(4)
        an_col1.metric("Noise Reduction", f"{anti_noise_stats['cancellation_db']:.2f} dB")
        an_col2.metric("Power Attenuation", f"{anti_noise_stats['attenuation_percent']:.1f}%")
        an_col3.metric("Reference Correlation", f"{anti_noise_stats['correlation']:.3f}")
        an_col4.metric("Phase Inversion Accuracy", f"{anti_noise_stats['phase_accuracy']:.1f}%")

        # 3. Render Spectrograms
        st.subheader("3. Spectrogram Evidence (Noisy vs. Filtered vs. Clean)")
        fig = render_spectrogram_panel(speech, primary, enhanced, fs)
        st.pyplot(fig)

        # Save & Display Audio Players
        os.makedirs("outputs", exist_ok=True)
        noisy_path = "outputs/noisy_input_temp.wav"
        enhanced_path = "outputs/enhanced_output_temp.wav"

        sf.write(noisy_path, primary, fs)
        sf.write(enhanced_path, enhanced, fs)

        # 4. Real-Time Audio Playback
        st.subheader("4. Real-Time Audio Playback")
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            st.markdown("**Noisy Primary Input Signal**")
            st.audio(noisy_path)

        with col_audio2:
            st.markdown("**Enhanced Output (Anti-Noise Applied)**")
            st.audio(enhanced_path)