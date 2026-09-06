# EV Cabin Active Noise Control (ANC)

A multi-stage digital signal processing and TinyML-based active noise control system for reducing noise inside electric vehicle (EV) cabins while maintaining speech intelligibility and real-time performance.

---

## Project Overview

Electric vehicles do not have the same engine noise as conventional vehicles, making other sounds inside the cabin more noticeable.

The main noise sources considered in this project include:

* High-frequency inverter and motor noise
* Continuous road and tire noise
* Sudden transient sounds such as pothole impacts and door slams
* Non-stationary residual noise

The system uses two microphone channels and combines real-time adaptive filtering with a TinyML-based control engine.

The overall processing architecture consists of:

* **Stage 0:** Kurtosis-based impulse pre-filter
* **Stage 1:** VAD-gated NLMS adaptive filtering
* **Stage 2:** Spectral Wiener post-filter
* **TinyML Control Engine:** Classifies acoustic conditions and provides VAD information to dynamically control the DSP pipeline

---

# System Architecture

The system receives audio from a primary microphone and a reference microphone.

```text
                         Raw Audio Inputs
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        Primary Microphone              Reference Microphone
        (Speech + Noise)                  (Ambient Noise)
                │                               │
                └───────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │ Stage 0: Impulse Pre-Filter  │
                 │ Kurtosis-based detection     │
                 └──────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       Real-Time DSP Loop              TinyML Control Engine
       Sample-by-Sample                 Async / Periodic
                │                               │
                ▼                               │
       ┌───────────────────┐                     │
       │ Stage 1: NLMS     │◄────────────────────┘
       │ Adaptive Filter   │       Control Parameters
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Stage 2: Spectral │
       │ Wiener Post-Filter│
       └─────────┬─────────┘
                 │
                 ▼
          Cleaned Speech Output
```

The TinyML engine operates as a separate control layer rather than replacing the real-time DSP processing.

---

# How the System Works

## Primary and Reference Microphones

The system uses two audio channels.

### Primary Microphone

```text
Primary = Speech + Noise
```

This is the main signal that needs to be cleaned.

### Reference Microphone

```text
Reference = Ambient Noise
```

The reference channel provides information about environmental noise and is used by the adaptive filtering stage.

---

# Stage 0 — Kurtosis-Based Impulse Pre-Filter

The first stage handles sudden, high-amplitude transient disturbances.

Examples include:

* Pothole impacts
* Door slams
* Other impulsive disturbances

Kurtosis is used to identify signal windows containing impulsive events.

Detected transient regions are repaired before entering the adaptive filtering stage. This reduces the possibility of large impulsive signals disturbing the adaptive filter.

Implementation:

```text
src/impulse.py
```

---

# Stage 1 — VAD-Gated NLMS Adaptive Filtering

The second stage uses a Normalized Least Mean Squares (NLMS) adaptive filter.

The reference microphone signal is used to estimate the noise component present in the primary microphone.

```text
Primary microphone
       │
       │ Speech + Noise
       ▼
    NLMS Filter ◄──── Reference microphone
       │
       ▼
    Residual/Error
```

The filter is **VAD-gated** so that adaptation can be controlled according to speech activity.

The current implementation uses a configurable adaptation step size and freezes adaptation during detected speech regions to reduce unnecessary modification of speech components.

Implementation:

```text
src/nlms.py
```

The current pipeline uses:

```text
NLMS taps = 128
μ = 0.03
```

---

# Stage 2 — Spectral Wiener Post-Filter

The residual signal from the adaptive filtering stage is processed using a spectral post-filter.

The post-filter operates in the frequency domain and suppresses residual noise while attempting to preserve important speech components.

The suppression strength is controlled through the spectral floor parameter.

Implementation:

```text
src/spectral_mask.py
```

The current pipeline uses a spectral floor gain of:

```text
floor_gain = 0.45
```

---

# End-to-End Processing Pipeline

The current DSP pipeline can therefore be summarized as:

```text
Primary Mic ───────────────┐
                           │
                           ▼
                    Stage 0
              Impulse Detection
                           │
                           ▼
                    Stage 1
                VAD-Gated NLMS
                           │
                           ▼
                    Stage 2
              Spectral Post-Filter
                           │
                           ▼
                 Enhanced Speech
```

The reference microphone provides the noise reference required by the NLMS stage.

---

# Datasets and Mixing

## Clean Speech

Clean speech samples are taken from the **LibriSpeech ASR Corpus**, using the `dev-clean` subset.

The speech recordings are sampled at:

```text
16 kHz
```

---

## EV Cabin Noise

The project uses custom automotive noise profiles representing different EV cabin conditions.

```text
ev_inverter_hum.wav
ev_road_tire_noise.wav
ev_pothole_thumps.wav
```

Additional stress-test noise recordings are also included:

```text
gunshots_fireworks.flac
helicopter.wav
```

### Noise Types

#### `ev_inverter_hum.wav`

Represents tonal/high-frequency noise associated with EV power electronics.

#### `ev_road_tire_noise.wav`

Represents continuous broadband road and tire noise.

#### `ev_pothole_thumps.wav`

Represents sudden transient disturbances caused by road impacts.

#### `helicopter.wav`

Used as a non-stationary stress-test noise source.

#### `gunshots_fireworks.flac`

Used as an impulsive stress-test noise source.

---

# Audio Mixing

The mixing process is handled by:

```text
src/data_prep.py
```

The system creates the signals required for the two-microphone ANC setup.

### Primary Channel

```text
Primary = Speech + Noise
```

The speech and noise are mixed at multiple SNR conditions:

```text
-3 dB
 0 dB
+3 dB
```

### Reference Channel

```text
Reference = Noise
```

The reference signal is passed to the adaptive filtering stage.

---

## Randomized Noise Slicing

Different sections of the noise recordings can be selected during sample generation.

This reduces dependence on a single fixed portion of a recording and provides more varied evaluation conditions.

---

# Cabin Acoustic Simulation

The project uses `pyroomacoustics` to simulate a simplified cabin acoustic environment.

The simulation models:

* Speech propagation to the primary microphone
* Noise propagation to the primary microphone
* Noise propagation to the reference microphone
* Room reverberation

Implementation:

```text
src/room_sim.py
```

The simulated microphone signals are used to construct the primary and reference channels for evaluation.

---

# Repository Structure

```text
adaptive_noise_cancellation/
│
├── data/
│   ├── clean_speech/          # LibriSpeech clean speech recordings
│   ├── mixed/                 # Generated scenario audio (ignored by Git)
│   └── noise/                 # EV and stress-test noise recordings
│
├── outputs/                   # Generated visualizations and audio outputs
│
├── src/
│   ├── __init__.py
│   ├── control_interface.py   # TinyML output → DSP parameter mapping
│   ├── data_prep.py           # Speech/noise mixing and scenario generation
│   ├── evaluate.py            # STOI and SNR evaluation
│   ├── impulse.py             # Stage 0 impulse detection and repair
│   ├── make_impulse.py        # Synthetic impulse generation
│   ├── nlms.py                # Stage 1 NLMS adaptive filter
│   ├── pipeline.py            # End-to-end DSP pipeline
│   ├── room_sim.py            # Acoustic propagation simulation
│   ├── spectral_mask.py       # Stage 2 spectral post-filter
│   └── vad_fast.py            # Fast energy-based VAD
│
├── build_samples.py           # Dataset/scenario generator
├── demo.py                    # Demonstration runner
├── fetch_ev_data.py           # EV noise data generation/downloader
├── validate.py                # Multi-SNR validation and ablation testing
├── visualize.py               # Waveform and spectrogram visualization
├── requirements.txt           # Python dependencies
└── .gitignore
```

Generated files such as mixed audio, intermediate outputs, and Python cache files are excluded from version control.

---

# Installation

## Requirements

The project uses Python and the following major libraries:

* NumPy — numerical operations
* SciPy — signal processing and kurtosis calculation
* PyRoomAcoustics — acoustic room simulation
* SoundFile — WAV/FLAC audio I/O
* librosa — audio processing utilities
* Matplotlib — visualization
* Pystoi — speech intelligibility evaluation

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Generate / Prepare Samples

Run:

```bash
python build_samples.py
```

This generates the mixed microphone scenarios used during validation and visualization.

Generated samples are stored locally in:

```text
data/mixed/
```

These generated files are excluded from Git.

---

## 2. Run Validation

Run:

```bash
python validate.py
```

The validation suite evaluates the pipeline across:

```text
-3 dB SNR
 0 dB SNR
+3 dB SNR
```

It compares the noisy input against the enhanced output using:

* STOI
* SNR improvement

The validation script also evaluates different pipeline configurations to examine the contribution of individual processing stages.

Results are written locally to:

```text
outputs/validation_results.csv
```

---

## 3. Generate Visualizations

Run:

```bash
python visualize.py
```

The visualization script generates waveform and spectrogram comparisons between:

1. Noisy input
2. Enhanced output
3. Clean speech reference

Generated visualizations are stored in:

```text
outputs/
```

---

# Evaluation Metrics

## STOI

**STOI (Short-Time Objective Intelligibility)** is used to evaluate speech intelligibility.

The score ranges approximately from:

```text
0 → 1
```

A higher score indicates better estimated speech intelligibility.

STOI is particularly useful for this project because the goal is not simply to reduce signal energy, but to reduce noise while preserving speech.

---

## SNR Gain

Signal-to-noise ratio gain measures the improvement in SNR after processing.

```text
SNR Gain = SNR After - SNR Before
```

A positive SNR gain indicates that the processed signal has a higher measured SNR than the input signal.

---

# Validation Strategy

The system is evaluated under multiple acoustic conditions rather than using a single noise recording.

The validation suite includes combinations of:

* EV cabin noise
* Inverter hum
* Road/tire noise
* Pothole impulses
* Helicopter noise
* Fireworks/gunshot impulsive noise
* Mixed noise conditions

Each condition is evaluated at:

```text
-3 dB
 0 dB
+3 dB
```

Multiple clean speech recordings are also used to reduce dependence on a single speech sample.

---

# TinyML Control Engine

The TinyML component is being developed as a **separate module by the TinyML team member**.

It does not replace the real-time DSP pipeline. Instead, it acts as an asynchronous control engine that provides information about the current acoustic environment.

The intended architecture is:

```text
                  TinyML Control Engine
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
            Noise Class            VAD State
                │                     │
                └──────────┬──────────┘
                           ▼
                Dynamic DSP Parameters
                           │
                           ▼
                Real-Time ANC Pipeline
```

The TinyML model communicates with the DSP system through:

```text
src/control_interface.py
```

---

## Model Output Interface

The TinyML model is expected to emit a JSON message on every approximately 100 ms hop:

```json
{
    "class": "stationary",
    "confidence": 0.85
}
```

Valid classes are:

```text
stationary
non_stationary
speech
```

The control interface applies confidence filtering before accepting a model prediction.

Current confidence threshold:

```text
confidence >= 0.60
```

Predictions below this threshold are ignored.

---

# Dynamic Suppression Profile

The intended TinyML-to-DSP mapping is:

| Detected Condition   | Intended System Response                                       |
| -------------------- | -------------------------------------------------------------- |
| Stationary Noise     | Increase NLMS adaptation and use stronger spectral suppression |
| Non-Stationary Noise | Use moderate adaptive filtering and stronger post-filtering    |
| Speech Detected      | Freeze NLMS adaptation to protect speech components            |

The exact control parameters can be tuned during TinyML/DSP integration.

---

## TinyML Class Design

The system intentionally uses three acoustic classes:

```text
stationary
non_stationary
speech
```

An `impulsive` TinyML class is **not required**.

Impulsive events such as pothole impacts are handled upstream by:

```text
src/impulse.py
```

using kurtosis-based transient detection.

This keeps the TinyML classifier focused on broader acoustic states while allowing the deterministic DSP stage to handle short impulsive events.

---

# TinyML Dataset Plan

The proposed TinyML training dataset consists of approximately:

```text
1,500–2,000 samples per class
```

using approximately:

```text
500 ms audio context
100 ms hop
```

Target classes and sources:

| Class            | Target Samples | Recommended Source                 |
| ---------------- | -------------: | ---------------------------------- |
| `stationary`     |    1,500–2,000 | DEMAND: NOFFICE, NFIELD, NPARK     |
| `non_stationary` |    1,500–2,000 | DEMAND: PCAFE, PSTATION, SSTRAFFIC |
| `speech`         |    1,500–2,000 | LibriSpeech `dev-clean`            |

---

## Train / Validation Split

Adjacent audio frames from the same recording are highly correlated.

Therefore, the dataset should **not** be split by randomly shuffling individual frames.

Instead:

* Hold out complete recordings for validation.
* For speech, hold out complete speakers where possible.
* Avoid placing adjacent frames from the same recording in both training and validation sets.

This prevents acoustic and speaker information from leaking between the training and validation sets.

---

# TinyML Model Constraints

The intended embedded classifier is designed for resource-constrained hardware.

Proposed features:

```text
13–20 MFCC coefficients
500 ms context window
100 ms hop
```

Candidate topology:

```text
1D CNN / Depthwise-Separable Conv1D
        ↓
Global Average Pooling
        ↓
Softmax
```

Target deployment constraints:

| Parameter             |               Target |
| --------------------- | -------------------: |
| Model weights         |             < 256 KB |
| SRAM / working memory |              < 64 KB |
| Inference latency     |              < 10 ms |
| Quantization          |                 INT8 |
| Target hardware       | ESP32-S3 / Cortex-M4 |

These values represent the intended deployment targets and can be refined after hardware profiling.

---

# Results

The validation framework is implemented and evaluates the pipeline across multiple scenarios and SNR conditions.

Because the pipeline is actively being tuned and expanded, **results in this section should be generated directly from the current `validate.py` output rather than relying on previously recorded benchmark values**.

Run:

```bash
python validate.py
```

to generate the latest:

```text
outputs/validation_results.csv
```

The main metrics reported are:

```text
STOI Before
STOI After
STOI Gain
SNR Before
SNR After
SNR Gain
```

This keeps the reported benchmark values synchronized with the current implementation.

---

# Spectrogram Analysis

The visualization pipeline provides a qualitative comparison of the signals.

### Noisy Input

Contains the speech signal together with the simulated cabin/environmental noise.

### Enhanced Output

Shows the signal after the ANC processing stages.

The expected effect is reduction of unwanted noise components while retaining important speech characteristics.

### Clean Reference

Provides the clean speech signal used as the ground-truth reference.

The three signals can therefore be compared in both the time and frequency domains.

---

# Current Project Status

### Implemented

* Two-microphone signal architecture
* Clean speech and noise mixing
* Multiple SNR conditions
* Randomized noise slicing
* Simplified cabin acoustic simulation
* Kurtosis-based transient detection
* VAD-gated NLMS adaptive filtering
* Spectral Wiener post-filter
* STOI evaluation
* SNR evaluation
* Multi-scenario validation
* Waveform and spectrogram visualization
* TinyML control interface architecture

### In Progress

* TinyML 3-class acoustic classifier
* Integration of TinyML predictions with DSP parameters
* Expanded acoustic dataset
* Embedded deployment optimization

---

# Future Work

## 1. TinyML Integration

Integrate the trained 3-class CNN with:

```text
src/control_interface.py
```

so that acoustic classification dynamically controls the DSP parameters.

---

## 2. Dataset Expansion

Add more real-world vehicle noise recordings and acoustic conditions.

Potential sources include:

* DEMAND Multichannel Acoustic Noise Database
* Freesound
* AudioSet
* Additional automotive recordings

---

## 3. Speech Dataset Expansion

Additional LibriSpeech samples can be used to evaluate performance across:

* More speakers
* Different speaking styles
* Different speech durations

---

## 4. Embedded Deployment

The eventual goal is to deploy the DSP + TinyML system on resource-constrained hardware such as:

* ESP32-S3
* ARM Cortex-M4 class microcontrollers
* Other suitable automotive/embedded DSP platforms

The TinyML model can be quantized to INT8 for deployment using frameworks such as:

* TensorFlow Lite for Microcontrollers
* STM32Cube.AI

---

## 5. Interactive Demonstration

A lightweight interface can eventually provide:

* Noisy audio playback
* ANC-enhanced audio playback
* TinyML-controlled output
* Waveform visualization
* STFT spectrograms
* STOI values
* SNR values
* Noise-class predictions
* VAD state

Possible frameworks include Streamlit or Gradio.

---

# Team Integration Workflow

The TinyML implementation should be developed on a separate feature branch without directly modifying `main`.

Example:

```bash
git checkout main
git pull origin main

git checkout -b feature/tinyml-cnn-integration
```

The TinyML implementation can be placed under a dedicated package, for example:

```text
src/tinyml/
├── __init__.py
├── inference.py
└── model.py
```

The model should communicate with the DSP system through:

```text
src/control_interface.py
```

After integration, the complete system can be tested using:

```bash
python validate.py
python visualize.py
```

and the TinyML implementation can then be committed and pushed through its own feature branch.

---

# Project Goal

The long-term goal is to develop a lightweight, real-time EV cabin ANC system that combines:

```text
Acoustic Simulation
        +
Impulse Detection
        +
Adaptive Noise Cancellation
        +
Spectral Noise Suppression
        +
TinyML Acoustic Classification
        +
Embedded Deployment
```

The system is designed to reduce cabin noise while maintaining speech intelligibility and remaining suitable for real-time embedded operation.
