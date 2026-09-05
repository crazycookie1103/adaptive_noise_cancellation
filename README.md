# EV Cabin Active Noise Control (ANC)

A multi-stage digital signal processing and TinyML-based active noise control system for reducing noise inside electric vehicle (EV) cabins while maintaining speech intelligibility and real-time performance.

---

## Project Overview

Electric vehicles do not have the same engine noise as conventional vehicles, which makes other sounds inside the cabin more noticeable.

The main noise sources considered in this project are:

* High-frequency inverter and motor noise
* Continuous road and tire noise
* Sudden transient sounds such as pothole impacts and door slams
* Non-stationary residual noise

The system uses two microphone channels and combines real-time adaptive filtering with a TinyML control engine.

The processing pipeline consists of:

* **Stage 0:** Kurtosis-based impulse pre-filter
* **Stage 1:** VSS-NLMS adaptive filtering
* **Stage 2:** Spectral Wiener post-filter
* **TinyML Control Engine:** Classifies the noise and provides VAD information to dynamically control the ANC pipeline

---

# System Architecture

The system receives audio from a primary microphone and a reference microphone.

The primary microphone contains both speech and noise, while the reference microphone captures the surrounding ambient noise.

```text
Raw Audio Inputs
       │
       ├──────────────────────────────────────────────┐
       │                                              │
       ▼                                              ▼
Primary Mic Channel                         Reference Mic Channel
(Speech + Noise)                              (Ambient Noise)
       │                                              │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │ Stage 0: Kurtosis Impulse Pre-Filter  │
              │ (Clamps transients / door slams)      │
              └──────────────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────────┐       ┌─────────────────────────────┐
          │ Real-Time Audio Loop  │       │ Async TinyML Control Engine │
          │   (Sample-by-Sample)  │       │   (Runs every 100-200 ms)   │
          └──────────┬────────────┘       └──────────────┬──────────────┘
                     │                                   │
                     ▼                                   │
          ┌───────────────────────┐                       │
          │ Stage 1: VSS-NLMS     │                       │
          │ Removes Stationary    │                       │
          │ Coherent Noise        │                       │
          └──────────┬────────────┘                       │
                     │                                   │
                     │ Residual Audio +                   │
                     │ Primary Error                      │
                     │                                   │
                     ▼                                   │
          ┌───────────────────────┐                       │
          │ Stage 2: Spectral     │◄──────────────────────┘
          │ Post-Filter (Wiener)  │
          │ Removes Uncorrelated  │
          │ Non-Stationary Residual│
          └──────────┬────────────┘
                     │
                     ▼
             Cleaned Speech Output
```

---

# How the System Works

## Primary and Reference Microphones

The system uses two audio channels.

### Primary Mic Channel

```text
Primary = Speech + Noise
```

This is the main audio signal that needs to be cleaned.

### Reference Mic Channel

```text
Reference = Ambient Noise
```

The reference channel provides information about the environmental noise and is used by the adaptive filtering stage.

---

## Stage 0 — Kurtosis Impulse Pre-Filter

The first stage handles sudden, high-amplitude transient sounds.

Examples include:

* Door slams
* Pothole impacts
* Other impulsive disturbances

Kurtosis is used to detect these transient events.

Once detected, the transient is clamped before it enters the main adaptive filtering stage.

This prevents large impulsive signals from disturbing the adaptive filter.

---

## Stage 1 — VSS-NLMS Adaptive Filtering

The real-time audio loop processes the signal sample-by-sample using a **Variable Step-Size Normalized Least Mean Squares (VSS-NLMS)** adaptive filter.

This stage primarily removes **stationary coherent noise** using the reference microphone signal.

The VSS-NLMS stage produces:

```text
Residual Audio + Primary Error
```

These signals are then passed to the spectral post-filter.

---

## Stage 2 — Spectral Post-Filter

The remaining audio is processed using a **Wiener spectral post-filter**.

This stage removes **uncorrelated non-stationary residual noise** that remains after the adaptive filtering stage.

The amount of suppression is controlled by the information provided by the TinyML control engine.

---



# Datasets and Mixing

## Clean Speech

Clean speech samples are taken from the **LibriSpeech ASR Corpus**, using the `dev-clean` subset.

The speech recordings are sampled at **16 kHz**.

## EV Cabin Noise

The project uses custom automotive noise profiles representing different EV cabin conditions.

```text
ev_inverter_hum.wav
ev_road_tire_noise.wav
ev_pothole_thumps.wav
```

### Noise Types

**`ev_inverter_hum.wav`**

High-frequency tonal noise produced by power electronics.

**`ev_road_tire_noise.wav`**

Continuous broadband noise caused by road and tire interaction.

**`ev_pothole_thumps.wav`**

Non-stationary transient sounds caused by sudden road impacts.

---

# Audio Mixing

The mixing process is handled by:

```text
src/data_prep.py
```

The script creates the two microphone channels required by the ANC system.

### Primary Channel

```text
Primary = Speech + Noise
```

The speech and noise are mixed at specific SNR values:

```text
-3 dB
 0 dB
+3 dB
```

### Reference Channel

```text
Reference = Noise
```

The reference noise signal is used by the VSS-NLMS adaptive filter.

### Randomized Noise Slicing

Different time windows are selected from the noise recordings during testing.

This prevents the evaluation from depending on one fixed section of a noise recording.

---

# Repository Structure

```text


## 1. Directory Tree


```text
anc-poc/
├── data/
│   ├── clean_speech/          # LibriSpeech clean audio subdirectories
│   ├── mixed/                 # Pre-mixed scenario cache
│   └── noise/                 # Inverter, tire, and road noise audio files
├── outputs/                   # Visualizations (.png) and test audio (.wav)
├── src/
│   ├── __init__.py
│   ├── control_interface.py   # State machine mapping CNN outputs -> DSP parameters
│   ├── data_prep.py           # Multi-channel scenario mixer & offset slicer
│   ├── evaluate.py            # STOI and SNR gain computation metrics
│   ├── impulse.py             # Stage 0: Transient impulse detection & repair
│   ├── make_impulse.py        # Synthetic impulse generator for stress tests
│   ├── nlms.py                # Stage 1: VAD-Gated NLMS adaptive filter
│   ├── pipeline.py            # End-to-end pipeline execution wrapper
│   ├── room_sim.py            # Cabin acoustic transfer function simulation
│   ├── spectral_mask.py       # Stage 3: Spectral Wiener post-filter mask
│   └── vad_fast.py            # Fast energy Voice Activity Detector
├── build_samples.py           # Sample dataset generator script
├── demo.py                    # Demonstration runner script
├── fetch_ev_data.py           # Open-source dataset downloader script
├── validate.py                # Multi-SNR stress-test and ablation suite
├── visualize.py               # 3-Panel STFT spectrogram generator
└── requirements.txt           # Environment dependencies
```

---

# Installation

## Dependencies

The project uses the following Python libraries:

* **NumPy** — numerical operations on audio signals
* **SciPy** — signal processing and kurtosis calculation
* **SoundFile** — reading and writing WAV/FLAC files
* **Pystoi** — STOI speech intelligibility measurement
* **Matplotlib** — spectrogram visualization

## Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt or run pip install numpy scipy pyroomacoustics soundfile librosa matplotlib pystoi
```

---

# Running the Project

## Run Validation

Run the complete validation script:

```bash
python validate.py
```

The validation script tests the system at:

```text
-3 dB SNR
 0 dB SNR
+3 dB SNR
```

It reports the speech intelligibility and SNR improvement before and after processing.

---

## Generate Spectrograms

Run:

```bash
python visualize.py
```

This generates a three-panel STFT comparison:

1. Noisy input
2. Enhanced output
3. Clean speech

The resulting image is saved as:

```text
outputs/anc_spectrogram_comparison.png
```

---

# Results

The current pipeline produces the following results:

```text
Configuration                       | SNR   | STOI In   | STOI Out  | Delta    | SNR Gain
------------------------------------------------------------------------------------------
Full Pipeline                       | -3.0  | 0.7087    | 0.9298    | +0.2211  | +7.95 dB
Full Pipeline                       | +0.0  | 0.7854    | 0.9363    | +0.1509  | +5.69 dB
Full Pipeline                       | +3.0  | 0.8525    | 0.9533    | +0.1008  | +3.60 dB
```

The system is tested across multiple SNR conditions to check whether the pipeline remains stable as the noise level changes.

---

# Evaluation Metrics

## STOI

**STOI (Short-Time Objective Intelligibility)** is used to measure speech intelligibility.

The score ranges from:

```text
0 → 1
```

A higher score indicates better speech intelligibility.

For the tested **-3 dB SNR** condition:

```text
STOI Before = 0.7087
STOI After  = 0.9298
```

This gives a STOI improvement of:

```text
+0.2211
```

---

## SNR Gain

SNR gain measures the change in signal-to-noise ratio after processing.

```text
SNR Gain = SNR After - SNR Before
```

The highest measured improvement in the current test results is:

```text
+7.95 dB
```

---

# Spectrogram Analysis

The visualization script generates a three-panel comparison.

### Panel 1 — Noisy Input

Shows the speech signal mixed with EV cabin noise.

### Panel 2 — Enhanced Output

Shows the result after the complete ANC pipeline.

The background noise is reduced while the main speech features are retained.

### Panel 3 — Clean Ground Truth

Shows the original clean speech signal used as the reference.

This allows the enhanced output to be compared directly with the clean signal.


---


# TinyML (IN SO FAR this is what the crux of the chats were)(follow what fits)

The TinyML model is an important part of the system.

It does not replace the real-time ANC processing. Instead, it works as an **asynchronous control engine** for the ANC pipeline.

The TinyML engine runs every **100–200 ms** and provides:

* **Noise Class**
* **VAD State**

These outputs are used to dynamically change how the real-time audio processing behaves.

```text
                    TinyML Control Engine
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
             Noise Class              VAD State
                 │                       │
                 └───────────┬───────────┘
                             ▼
                  Dynamic Suppression
                       Profile
                             │
                             ▼
                  Real-Time ANC Pipeline
```

## Dynamic Suppression Profile

The system changes its processing according to the detected condition.

| Detected Condition   | System Response               |
| -------------------- | ----------------------------- |
| Stationary Noise     | Maximize NLMS μ               |
| Non-Stationary Noise | Boost post-filter attenuation |
| Speech Detected      | Freeze NLMS (VAD)             |

This allows the system to respond differently to different noise conditions instead of using the same filtering settings all the time.

---
```text
          
[ TinyML CNN / Stub File ] 
        │  Emits JSON every 100ms
        │  {"class": "stationary", "confidence": 0.85}
        ▼
[ src/control_interface.py ]  <-- PERMANENT BRIDGE
        │  1. Checks confidence (>= 0.6)
        │  2. Maps class to target parameters
        │  3. Smooths transition via EMA (step_smooth)
        ▼
[ NLMS Filter & Wiener Mask ]  <-- DSP Pipeline
        Uses smoothed mu & floor_gain

## 3. specs u can consider

### Model Output Interface Contract

Your model implementation must emit JSON messages matching this schema on every **100 ms hop**:

```json
{
  "class": "stationary",
  "confidence": 0.85
}
```

**Valid `class` strings:** `"stationary"`, `"non_stationary"`, `"speech"`.

**Confidence filtering:** Messages with `confidence < 0.60` are ignored by `ControlState.update()`.

### Target Class Mapping & Exclusions

* **`stationary`**: Maps to `μ = 0.50`, `floor_gain = 0.15` (Aggressive filtering for inverter hum/HVAC).
* **`non_stationary`**: Maps to `μ = 0.20`, `floor_gain = 0.35` (Moderate adaptation for road/tire rumble).
* **`speech`**: Maps to `μ = 0.05`, `floor_gain = 0.60` (Freezes NLMS weights to preserve voice formants).
* **CRITICAL RULE:** **Do NOT create an `impulsive` class.** Non-stationary impulsive thumps (potholes) are stripped upstream by Stage 0 (`src/impulse.py`) using Kurtosis filtering before reaching this layer.

---

## YOU CAN DO THIS SUGGESTION AGAIN

To train the 3-class model, gather **1,500–2,000 samples (500 ms window context)** per class (**~12.5–16.5 minutes** raw audio per class, **~50 minutes** total).

### Train / Validation Split Rule (CRITICAL)

**Do NOT randomly shuffle frame samples across train/validation splits.** Adjacent 500 ms frames from the same recording share acoustic profiles and speaker characteristics. Shuffling frame samples will leak data into validation sets, inflating metrics while failing in real-world deployment.

* **Partition Rule:** Hold out *entire audio files and speakers* when building train/validation sets.

| **Class**            | **Target Samples** | **Raw Duration** | **Recommended Source**                    | **Split Strategy**     |
| -------------------- | -----------------: | ---------------: | ----------------------------------------- | ---------------------- |
| **`stationary`**     |      1,500 – 2,000 |  12.5 – 16.5 min | DEMAND (`NOFFICE`, `NFIELD`, `NPARK`)     | Hold out full files    |
| **`non_stationary`** |      1,500 – 2,000 |  12.5 – 16.5 min | DEMAND (`PCAFE`, `PSTATION`, `SSTRAFFIC`) | Hold out full files    |
| **`speech`**         |      1,500 – 2,000 |  12.5 – 16.5 min | LibriSpeech (`dev-clean` subset)          | Hold out full speakers |

---

## 5. Model Footprint & Execution Constraints

* **Features:** 13–20 MFCC coefficients over a **500 ms** window (sliding with **100 ms** hop size).
* **Topology:** 1D CNN / Depthwise Separable Conv1D + Global Average Pooling + Softmax.
* **Binary Footprint:** **30 KB – 50 KB** max (INT8 quantized via TensorFlow Lite for Microcontrollers).
* **SRAM / Working Memory:** **< 64 KB** peak RAM consumption.
* **Inference Latency:** **< 20 ms execution per 100 ms frame hop** on target hardware (ESP32-S3 / Cortex-M4), leaving **80 ms CPU headroom** for NLMS filtering and spectral masking.

---

## 6. Teammate Integration Workflow

Create a new feature branch to implement your model under a stub/wrapper module without modifying the `main` branch:

```bash
# 1. Fetch latest main and check out feature branch
git checkout main
git pull origin main
git checkout -b feature/tinyml-cnn-integration

# 2. Place model code inside a dedicated package under src/
# e.g., src/tinyml/inference.py

# 3. Verify locally with validation and visualization scripts
python validate.py
python visualize.py

# 4. Push branch and open a Pull Request (PR) on GitHub
git add .
git commit -m "feat(tinyml): integrate 3-class 1D-CNN classifier with control_interface"
git push origin feature/tinyml-cnn-integration
```




The target limits for the TinyML model are(if scaling to hardware)

| Parameter              |              Target |
| ---------------------- | ------------------: |
| Flash Memory (Weights) |            < 256 KB |
| SRAM / Working RAM     |             < 64 KB |
| Inference Latency      |             < 10 ms |
| STOI Gain              | ≥ +0.15 at 0 dB SNR |

The model is intended to run on short audio frames while maintaining the required real-time response.

---

# TinyML Deployment(suggestion not final claude)

For embedded deployment, the CNN can be quantized to **8-bit integers (INT8)**.

Possible deployment frameworks include:

* TensorFlow Lite for Microcontrollers
* STM32Cube.AI

The goal is to keep the model small enough to run within the available Flash and RAM while maintaining the required inference speed.

---


## 6. Teammate Integration Workflow

Create a new feature branch to implement your model under a stub/wrapper module without modifying the `main` branch:

```bash
# 1. Fetch latest main and check out feature branch
git checkout main
git pull origin main
git checkout -b feature/tinyml-cnn-integration

# 2. Place model code inside a dedicated package under src/
# e.g., src/tinyml/inference.py

# 3. Verify locally with validation and visualization scripts
python validate.py
python visualize.py

# 4. Push branch and open a Pull Request (PR) on GitHub
git add .
git commit -m "feat(tinyml): integrate 3-class 1D-CNN classifier with control_interface"
git push origin feature/tinyml-cnn-integration
```

# Future Work

## 1. Dataset Expansion

Add more real-world vehicle noise recordings to test the system under a wider range of conditions.

Possible sources include:

* DEMAND Multichannel Acoustic Noise Database( we have used this)
* Freesound
* AudioSet

## 2. Speech Dataset Expansion

Additional LibriSpeech(used this) samples can be used to test the system with more speakers and different speech recordings.

## 3. Interactive Demo

A lightweight interface can be built using Streamlit or Gradio.

The demo can provide:

* Noisy audio playback
* Classical DSP output
* TinyML-controlled output
* Waveform visualization
* STFT spectrograms
* STOI values
* SNR values



