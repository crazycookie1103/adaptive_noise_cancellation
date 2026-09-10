# Adaptive Noise Cancellation System for Defence Applications

A real-time, multi-stage **Adaptive Noise Cancellation (ANC)** system designed for noisy communication environments, with particular focus on **road and EV acoustic noise** and potential defence applications. The system combines adaptive digital signal processing with a lightweight TinyML control engine to suppress environmental noise while preserving speech intelligibility.

The proposed embedded platform uses an **ESP32-S3** with dual I2S MEMS microphones for feedforward ANC. The current software pipeline is developed and evaluated primarily using road/EV and environmental noise data, while the architecture can be extended to impulsive and defence-specific acoustic conditions.

## 1. Project Information

- **Project Title:** Adaptive Noise Cancellation System for Defence Applications
- **PS ID:** `SIH26052`
- **PS Title:** `To Develop an AI/ML based -enabled adaptive noise cancellation system (ANC) that effectively surpresses stationary , non-stationary and impulsive defence noises while maintaining high speech intelligibility and real-performance on embedded hardware.`
- **Category:** Hardware
- **Theme:** Defence / Security

## 2. Problem Statement

Speech communication in vehicles and other high-noise environments can be degraded by stationary, non-stationary and sudden acoustic disturbances. Road and EV environments can contain persistent background noise, tyre/road noise, mechanical components and sudden transient events that vary over time.

The challenge is to develop an adaptive noise-control system that can respond to changing acoustic conditions, suppress unwanted noise and preserve speech intelligibility in real time. The same approach can be extended to high-noise defence communication environments.

## 3. Proposed Solution

The system uses two synchronized digital MEMS microphones:

- **Primary/Talk Microphone:** captures speech along with environmental noise.
- **Reference Microphone:** captures an ambient-noise reference used by the adaptive filtering stage.

The audio is processed through a multi-stage pipeline:

1. **Transient / Impulse Detection & Protection**
2. **VSS-NLMS Adaptive Filtering**
3. **Spectral Wiener Post-Filtering**
4. **TinyML-based Acoustic Classification and DSP Control**

The TinyML engine acts as an asynchronous control layer. Instead of using a single binary noise/speech toggle, the CNN produces three continuous confidence scores:

```text
p_stationary + p_non_stationary + p_speech = 1.0
```

These scores dynamically control the DSP parameters according to the current acoustic condition.

## 4. Key Features

- Real-time adaptive noise cancellation
- Dual-channel I2S digital audio acquisition
- Primary + ambient reference microphone architecture
- VSS-NLMS adaptive filtering for coherent environmental noise
- Transient / impulse detection and protection
- Wiener spectral post-filter for residual noise
- Three-class TinyML acoustic classifier
- Continuous confidence scores for stationary, non-stationary and speech conditions
- Dynamic DSP parameter control
- Speech-protection voice-band passthrough mask
- Objective evaluation using STOI and SNR improvement
- ESP32-S3-oriented embedded architecture

## 5. Technology Stack

- **Programming:** Python, C/C++ for embedded implementation
- **DSP / Audio:** NumPy, SciPy, Librosa, SoundFile, PyRoomAcoustics
- **Evaluation / Visualization:** Pystoi, Matplotlib
- **Machine Learning:** TensorFlow, Keras (`tf.keras`)
- **TinyML Model:** One lightweight CNN classifier using `tf.keras.Model` and `SeparableConv2D` layers
- **Model Output:** Three-class Softmax confidence vector — `p_stationary`, `p_non_stationary`, `p_speech`
- **Embedded Platform:** ESP32-S3, I2S, DMA
- **Audio Hardware:** Dual I2S MEMS microphones, MAX98357A + speaker for V1 demonstration

TensorFlow and Keras are both used in the TinyML workflow. **Keras (`tf.keras`)** is the high-level API used to construct, compile and train the CNN, while **TensorFlow** provides the underlying tensor/runtime operations and model conversion or quantization workflow.

## 6. Architecture

The overall processing flow is:

```text
Primary/Talk Mic ──> Speech + Noise ──┐
                                      │
Reference Mic ──> Ambient Noise ──────┤
                                      v
                         Transient / Impulse
                         Detection & Protection
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    VSS-NLMS              TinyML CNN Control
                         |                         |
                         |              p_stationary
                         |              p_non_stationary
                         |              p_speech
                         |                         |
                         +------------+------------+
                                      |
                                      v
                           Wiener Spectral Filter
                                      |
                                      v
                               Enhanced Speech
```

The current V1 design is a **feedforward ANC architecture** using two microphones. A future V2 configuration can add an internal error microphone and FxNLMS for closed-loop ANC.

### TinyML Control Logic

The main TinyML control model is a **lightweight three-class CNN** built with TensorFlow/Keras. It does not act as the primary noise-cancellation filter; it predicts the acoustic condition and controls the DSP stages.

The CNN produces three continuous confidence scores:

```text
p_stationary
p_non_stationary
p_speech
```

with:

```text
p_stationary + p_non_stationary + p_speech = 1.0
```

The control interface uses these values as follows:

| Confidence | DSP Control |
|---|---|
| `p_stationary` | Sets the Stage 1 VSS-NLMS step size `μ` and keeps Wiener over-subtraction low (`α ≈ 1.0`) |
| `p_non_stationary` | Boosts Stage 2 Wiener over-subtraction (`α ≈ 2.0–2.5`) for stronger post-filtering |
| `p_speech` | Activates the voice-band passthrough mask when speech confidence exceeds the custom `0.75` threshold |

This makes the TinyML model a **three-class CNN control engine**, rather than a single binary classifier or three separate networks.

## 7. How the System Works

### Primary and Reference Microphones

The primary microphone receives:

```text
d(n) = s(n) + v(n)
```

where:

- `s(n)` = desired speech
- `v(n)` = unwanted environmental noise

The reference microphone provides an estimate of the ambient noise:

```text
x(n) ≈ v(n)
```

The adaptive filter uses this reference to estimate the noise component present in the primary channel.

```text
y(n) = v̂(n)

e(n) = d(n) − y(n)
```

The target is:

```text
e(n) ≈ s(n)
```

The reference microphone should be described as an **Ambient Noise Reference**, rather than as capturing "noise only", because real acoustic environments may introduce speech, reflections, and other signals into the reference channel.

## 8. DSP Pipeline

### Stage 0 — Transient / Impulse Detection & Protection

The first stage detects sudden, high-amplitude acoustic events using kurtosis-based analysis.

Potential disturbances include:

- Impulsive mechanical sounds
- Sudden environmental impacts
- Other short-duration acoustic transients

Rather than simply clipping the signal, the detector protects the adaptive filter by controlling its behaviour during an impulse, for example by reducing or freezing adaptive updates.

```text
Audio
  |
  v
Transient Detector
  |
  v
Impulse Detected?
  |
  +---- YES ----> Reduce / Freeze Adaptive Update
  |
  +---- NO -----> Normal Adaptive Processing
```

### Stage 1 — VSS-NLMS Adaptive Filtering

The real-time audio path uses a **Variable Step-Size Normalized Least Mean Squares (VSS-NLMS)** adaptive filter.

This stage primarily suppresses coherent environmental noise using the reference microphone.

The adaptive output is passed to the following spectral stage as residual/error audio.

### Stage 2 — Wiener Spectral Post-Filter

The remaining signal is processed using a **Wiener spectral post-filter**.

This stage targets residual, uncorrelated, and non-stationary noise that remains after adaptive filtering.

The suppression profile can be dynamically adjusted by the TinyML control engine.

## 9. TinyML Control Engine

TinyML does not replace the real-time ANC signal-processing path. It acts as an **asynchronous control engine**.

The intended control flow is:

```text
                 TinyML Control Engine
                         |
               +----------+----------+
               |                     |
               v                     v
          Noise Class             VAD State
               |                     |
               +----------+----------+
                         |
                         v
                Dynamic DSP Profile
                         |
                         v
                 Real-Time ANC
```

The TinyML engine provides:

- Noise classification
- Voice Activity Detection (VAD)
- Dynamic control of adaptive-filter parameters
- Dynamic control of post-filter attenuation

### Confidence-Based DSP Control

Rather than a single binary classification, the model produces three continuous confidence scores:

```text
p_stationary
p_non_stationary
p_speech
```

These values are used by the decision logic to dynamically control the DSP parameters:

| Confidence | System Response |
|---|---|
| `p_stationary` | Sets Stage 1 VSS-NLMS step size `μ` and keeps Stage 2 Wiener over-subtraction low (`α ≈ 1.0`) |
| `p_non_stationary` | Boosts Stage 2 Wiener over-subtraction (`α ≈ 2.0–2.5`) for aggressive post-filtering |
| `p_speech` | Triggers the voice-band passthrough mask when speech confidence crosses the custom `0.75` threshold |

The model therefore functions as a **three-class CNN control engine**, rather than a single binary classifier or three separate networks.

## 10. Datasets and Audio Preparation

### Clean Speech

Clean speech samples are taken from the **LibriSpeech ASR Corpus**, using the `dev-clean` subset.

The speech recordings are sampled at **16 kHz**.

### Environmental Noise

The development and evaluation pipeline uses noise recordings from the **DEMAND Multichannel Acoustic Noise Database** and corresponding prepared noise scenarios.

The current development and evaluation focus is primarily on **road, tyre, traffic and EV-related environmental noise**, while the same pipeline can be extended to defence-specific acoustic conditions.

The dataset categories used for TinyML development include:

- Stationary noise environments such as `NOFFICE`, `NFIELD`, and `NPARK`
- Non-stationary environments such as `PCAFE`, `PSTATION`, and `SSTRAFFIC`

### Audio Mixing

The primary channel is formed as:

```text
Primary = Speech + Noise
```

The reference channel is:

```text
Reference = Noise Reference
```

The evaluation pipeline tests multiple input SNR conditions:

```text
-3 dB
 0 dB
+3 dB
```

Different time windows can be selected from noise recordings to avoid evaluating the system on only one fixed portion of a recording.

## 11. Hardware Components

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32-S3 | 1 | Real-time DSP, I2S acquisition, TinyML inference and control |
| Primary/Talk I2S MEMS Mic | 1 | Captures speech + environmental noise |
| Reference I2S MEMS Mic | 1 | Provides ambient-noise reference |
| MAX98357A | 1 | I2S DAC + Class-D amplifier for V1 speaker demonstration |
| 4–8 Ω Speaker | 1 | Acoustic output for V1 demonstration |
| SSD1306 OLED | 1 optional | Displays mode, noise class and telemetry |
| USB-C 5 V supply | 1 | Prototype power source |
| AMS1117-3.3 | 1 for V1 | 5 V to 3.3 V regulation |
| Efficient Buck Regulator | Future | Preferred for final battery-powered design |
| Internal Error MEMS Mic | 1 optional, V2 | Feedback microphone for FxNLMS |
| FTDI + 6-pin header | 1 | External programming interface |

### Why ESP32-S3?

The ESP32-S3 is suitable for the prototype because it provides:

- Dual-core processing
- Up to 240 MHz operation
- I2S peripheral for digital audio
- DMA support for continuous audio streaming
- Sufficient processing capability for NLMS/FxNLMS
- Lightweight TinyML capability
- Low-cost embedded implementation
- C/C++ development through ESP-IDF

The intended task split is:

```text
Core 0                         Core 1
Real-Time Audio                TinyML Control
     |                              |
I2S + DMA                     Audio context
     |                              |
Transient detection           MFCC / log-mel
     |                              |
VSS-NLMS / FxNLMS             Classification
     |                              |
Wiener filter                 Parameter update
     |                              |
I2S output                    ---> Core 0
```

This separation keeps the slower control/inference workload from blocking the hard real-time audio path.

## 12. Prototype Power Budget

The V1 hardware power budget is approximately:

| Load | Estimated Power |
|---|---:|
| ESP32-S3 | ~1.32 W |
| 2 × MEMS microphones | ~0.013 W |
| SSD1306 OLED | ~0.066 W |
| MAX98357A + speaker (typical) | ~1.11 W |
| **Estimated Total** | **~2.5 W** |

A design margin of approximately 50% gives a design requirement of about **3.75 W**.

Recommended prototype supply:

```text
5 V / 2 A minimum
5 V / 3 A preferred
```

The V1 prototype uses a USB-C 5 V supply. The MAX98357A is powered directly from the 5 V rail, while the ESP32-S3, microphones and OLED use the 3.3 V rail.

For the final battery-powered design, an efficient switching/buck regulator is preferred over the AMS1117-3.3 because the linear regulator can dissipate significant heat while converting 5 V to 3.3 V.

## 13. Hardware Connections

### I2S Microphones

Typical digital MEMS microphone connections:

```text
MEMS MIC             ESP32-S3
--------             --------
VDD       ----------> 3.3 V
GND       ----------> GND
SCK/BCLK  ----------> I2S BCLK
WS/LRCLK  ----------> I2S WS
DOUT      ----------> I2S DATA IN
L/R       ----------> Channel Selection
```

For the two-channel V1 architecture:

```text
Primary DOUT   ---> Left I2S channel
Reference DOUT ---> Right I2S channel

BCLK -----------+
WS/LRCLK -------+----> ESP32-S3 I2S RX
```

The exact microphone channel-selection configuration and ESP32-S3 GPIO assignments depend on the selected development board and microphone breakout.

### Audio Output

```text
ESP32-S3 I2S TX
      |
      +---- BCLK ----> MAX98357A BCLK
      +---- WS ------> MAX98357A LRCLK
      +---- DATA ----> MAX98357A DIN
                         |
                         v
                       Speaker
```

The MAX98357A is intended for the V1 **speaker demonstration**. A final headphone implementation would use an appropriate audio codec/headphone amplifier and headphone driver instead.

### OLED

```text
ESP32-S3
   |
   +---- SDA ----> OLED SDA
   +---- SCL ----> OLED SCL
```

The OLED is optional and is intended for demonstration/debugging rather than as part of the ANC signal path.

## 14. Repository Structure

```text
anc-poc/
├── data/
│   ├── clean_speech/        # LibriSpeech clean audio
│   ├── mixed/               # Pre-mixed scenario cache
│   └── noise/               # Inverter, tyre, road and EV noise audio
├── outputs/                 # Visualizations (.png) and test audio (.wav)
├── src/
│   ├── __init__.py
│   ├── control_interface.py # State-machine mapping CNN outputs -> DSP parameters
│   ├── data_prep.py         # Multi-channel scenario mixer and offset handling
│   ├── evaluate.py          # STOI and SNR-gain computation
│   ├── impulse.py           # Transient impulse detection and protection
│   ├── make_impulse.py      # Synthetic impulse generator for stress tests
│   ├── nlms.py              # VSS-NLMS adaptive filtering
│   ├── pipeline.py          # End-to-end pipeline execution
│   ├── room_sim.py          # Acoustic transfer-function simulation
│   ├── spectral_mask.py     # Wiener spectral post-filter
│   └── vad_fast.py          # Fast energy-based Voice Activity Detector
├── build_samples.py         # Sample dataset generator
├── demo.py                  # Demonstration runner
├── fetch_ev_data.py         # Open-source EV dataset downloader
├── validate.py              # Multi-SNR stress-test and ablation suite
├── visualize.py             # Three-panel STFT spectrogram generator
└── requirements.txt         # Environment dependencies
```

## 15. Final Presentation

Keep the final SIH presentation with the project submission materials whenever the file size allows it.

The presentation should summarize the problem, proposed ANC pipeline, TinyML control approach, system architecture, evaluation results and future hardware implementation.

## 16. Demo Video

A demo video is **optional, but recommended**.

The demonstration should show the ANC pipeline running on representative road/EV and environmental noise conditions, including the noisy input, enhanced output and objective evaluation results.

## 17. Screenshots / Prototype Photos

Add important screenshots, spectrograms, result plots or hardware/prototype photographs to the repository as they become available.

Useful demonstrations include:

- Noisy versus enhanced waveform
- STFT spectrogram comparison
- STOI improvement
- SNR improvement
- TinyML confidence outputs
- DSP parameter adaptation

## 18. Installation

```bash
git clone https://github.com/crazycookie1103/adaptive_noise_cancellation.git
cd adaptive_noise_cancellation
pip install -r requirements.txt
```

A Python virtual environment is recommended:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## 19. Run

### Run the validation suite

```bash
python validate.py
```

The validation pipeline evaluates the ANC system across multiple input SNR conditions:

```text
-3 dB
 0 dB
+3 dB
```

It reports speech-intelligibility and SNR improvements before and after processing.

### Generate spectrograms

```bash
python visualize.py
```

This generates a three-panel STFT comparison of:

1. Noisy input
2. Enhanced output
3. Clean speech reference

### Interactive Demo

If the Streamlit interface is present in the repository:

```bash
streamlit run app.py
```

The demo can present:

- Noisy audio
- Processed ANC output
- TinyML-controlled output
- Waveforms
- STFT spectrograms
- STOI values
- SNR values

### Run the demonstration

```bash
python demo.py
```

The demonstration runs the processing pipeline on prepared audio scenarios and produces processed audio and evaluation outputs.

### Current Evaluation Summary

For the three tested input conditions (`-3 dB`, `0 dB`, and `+3 dB`), the measured SNR gains are `+7.95 dB`, `+5.69 dB`, and `+3.60 dB`, respectively.

**Average measured SNR gain: `+5.75 dB`**

The pipeline also improves STOI across all three tested conditions, indicating improved speech intelligibility after processing.

## 20. Results

The current software evaluation pipeline reports the following results:

| Configuration | Input SNR | STOI In | STOI Out | STOI Δ | SNR Gain |
|---|---:|---:|---:|---:|---:|
| Full Pipeline | -3.0 dB | 0.7087 | 0.9298 | +0.2211 | +7.95 dB |
| Full Pipeline | 0.0 dB | 0.7854 | 0.9363 | +0.1509 | +5.69 dB |
| Full Pipeline | +3.0 dB | 0.8525 | 0.9533 | +0.1008 | +3.60 dB |

**Average measured SNR gain: `+5.75 dB`**

These results demonstrate improvement in speech intelligibility and signal-to-noise ratio across multiple input-noise conditions.

## 21. Evaluation Metrics

### STOI

**Short-Time Objective Intelligibility (STOI)** is used to evaluate speech intelligibility.

The score ranges from:

```text
0 → 1
```

A higher score indicates better speech intelligibility.

For the tested -3 dB SNR condition:

```text
STOI Before = 0.7087
STOI After  = 0.9298
Improvement = +0.2211
```

### SNR Gain

SNR gain is calculated as:

```text
SNR Gain = SNR After - SNR Before
```

Across the three tested input conditions, the average measured SNR gain is:

```text
+5.75 dB
```

## 22. Future Scope

- **Closed-loop ANC:** Add an internal error microphone and implement FxNLMS for feedback ANC.
- **Defence-specific dataset expansion:** Extend the current road/EV and environmental-noise evaluation with representative defence acoustic conditions, including impulsive events.
- **Embedded TinyML deployment:** Deploy the trained CNN control model on the ESP32-S3 using an embedded inference framework.
- **Improved adaptive control:** Further tune the continuous confidence-based control of NLMS step size, Wiener over-subtraction and speech protection.
- **Headset / protective-headgear integration:** Replace the V1 speaker demonstration with an appropriate headphone/earcup audio driver.
- **Real-time hardware validation:** Measure latency, attenuation, speech intelligibility and stability on physical hardware.

## Important

Before submission, make sure the repository is accessible to reviewers. Do **not** upload passwords, API keys, access tokens, `.env` files containing secrets, or other confidential credentials.
