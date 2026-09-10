
# Adaptive Noise Cancellation System for Defence Applications

A real-time, multi-stage **Adaptive Noise Cancellation (ANC)** system designed for defence communication and EV vehicles. The system combines adaptive digital signal processing with a lightweight TinyML control engine to suppress environmental noise while preserving speech intelligibility.

The proposed hardware uses an **ESP32-S3** as the central processing platform, with dual I2S MEMS microphones for feedforward ANC. A future V2 configuration adds an internal error microphone for closed-loop feedback ANC using FxNLMS.

---

## 1. Project Information

- **Project Title:** Adaptive Noise Cancellation System for Defence Applications
- **PS ID:** `SIH26052`
- **PS Title:** `To Develop an AI/ML based -enabled adaptive noise cancellation system (ANC) that effectively surpresses stationary , non-stationary and impulsive defence noises while maintaining high speech intelligibility and real-performance on embedded hardware.`
- **Category:** Hardware
- **Theme:** Defence / Security

---

## 2. Problem Statement

Defence personnel operating in high-noise environments may need to communicate clearly while being exposed to continuous and rapidly changing acoustic disturbances. Conventional passive hearing protection can reduce overall sound levels but may also make speech and important acoustic information harder to perceive.

The challenge is to develop an adaptive noise-control system that can identify changing noise conditions, suppress unwanted acoustic components, and preserve speech intelligibility in real time.

The system is intended as a **defence-oriented ANC platform**, with potential applications in communication headsets, protective headgear, vehicle crew systems, and other high-noise operational environments.

---

## 3. Proposed Solution

The proposed system uses two synchronized digital MEMS microphones:

- **Primary/Talk Microphone:** captures speech along with environmental noise.
- **Reference Microphone:** captures an ambient-noise reference used by the adaptive filtering stage.

The audio is processed on an **ESP32-S3** using a multi-stage pipeline:

1. **Transient / Impulse Detection & Protection**
2. **VSS-NLMS Adaptive Filtering**
3. **Spectral Wiener Post-Filtering**
4. **TinyML-based Noise Classification and VAD Control**

The TinyML engine operates asynchronously and dynamically adjusts the DSP parameters according to the detected acoustic condition.

The current V1 architecture is a **feedforward ANC configuration**. V2 extends the system with an internal error microphone for feedback/closed-loop ANC and FxNLMS.

---

## 4. Key Features

- Real-time adaptive noise cancellation
- Dual-channel I2S digital audio acquisition
- Primary + ambient reference microphone architecture
- VSS-NLMS adaptive filtering for coherent noise
- Transient / impulse detection and protection
- Wiener spectral post-filter for residual noise
- TinyML-based noise classification
- Voice Activity Detection (VAD)
- Dynamic DSP parameter control
- Dual-core ESP32-S3 processing architecture
- DMA-based continuous audio streaming
- OLED-based system telemetry for demonstration/debugging
- V1 feedforward ANC and V2 feedback ANC architecture
- Objective evaluation using STOI and SNR improvement

---

## 5. Technology Stack

### Hardware

- ESP32-S3
- INMP441 / equivalent I2S MEMS microphones
- MAX98357A I2S DAC + Class-D amplifier
- 4–8 Ω speaker for V1 demonstration
- SSD1306 128×64 OLED (optional)
- USB-C 5 V power input
- AMS1117-3.3 for the V1 prototype / efficient buck regulator recommended for the final battery-powered design
- External CP2102/FTDI programming interface
- Tkinter (for desktop GUI hardware simulation)

### Software / DSP

- Python
- NumPy
- SciPy
- Librosa
- SoundFile
- PyRoomAcoustics
- Matplotlib
- Pystoi
- Streamlit (for interactive demonstration, if enabled)

### Machine Learning

- MFCC / log-mel audio features
- Lightweight 1D CNN / Depthwise Separable Conv1D
- Global Average Pooling
- Softmax classification
- INT8 quantization for embedded deployment
- TensorFlow Lite for Microcontrollers (planned embedded deployment)

---

## 6. Architecture

### V1 — Feedforward ANC

```text
                 USB-C 5 V
                     |
          +----------+----------+
          |                     |
          v                     v
     3.3 V Regulator       MAX98357A
          |                     |
     +----+----+                v
     |    |    |             Speaker
     v    v    v
  ESP32  Mic  OLED
    -S3
     |
     | I2S + DMA
     |
 +---+-------------------+
 |                      |
 v                      v
Primary/Talk Mic    Reference Mic
Speech + Noise      Ambient Noise Reference
 |                      |
 +----------+-----------+
            |
            v
  Transient / Impulse
  Detection & Protection
            |
      +-----+------+
      |            |
      v            v
 Real-Time DSP   TinyML Control
      |            |
      v            v
 VSS-NLMS      Noise Class +
      |         VAD State
      |            |
      +-----<------+
            |
            v
   Wiener Spectral
    Post-Filter
            |
            v
       I2S TX
            |
            v
       MAX98357A
            |
            v
         Speaker
```

### V2 — Feedback ANC Extension

V2 adds a third **internal error microphone** positioned inside the earcup / near the protected listening region.

```text
Reference Mic ───────┐
                     |
Primary Mic ─────────┼──> ESP32-S3 ──> FxNLMS ──> Audio Output
                     |                    ^
                     |                    |
Internal Error Mic ──┴────────────────────┘
```

V1 uses two microphones for feedforward adaptive noise reduction. V2 adds the internal error microphone for true feedback/closed-loop ANC and FxNLMS.

---

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

---

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

---

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

### Noise-Class Mapping

| Detected Condition | System Response |
|---|---|
| Stationary Noise | Increase / maximize NLMS adaptation |
| Non-Stationary Noise | Increase post-filter attenuation |
| Speech Detected | Freeze NLMS adaptation to preserve speech |

The model interface can use messages of the following form:

```json
{
  "class": "stationary",
  "confidence": 0.85
}
```

Valid classes:

- `stationary`
- `non_stationary`
- `speech`

Low-confidence predictions below the configured confidence threshold can be ignored by the control interface.

---

## 10. Datasets and Audio Preparation

### Clean Speech

Clean speech samples are taken from the **LibriSpeech ASR Corpus**, using the `dev-clean` subset.

The speech recordings are sampled at **16 kHz**.

### Environmental Noise

The development and evaluation pipeline uses noise recordings from the **DEMAND Multichannel Acoustic Noise Database** and corresponding prepared noise scenarios.

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

---

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
| CP2102/FTDI + 6-pin header | 1 | External programming interface |

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

---

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

---

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

---

## 14. Repository Structure

```text
ADAPTIVE-ANC/
├── README.md
├── SUBMISSION_GUIDE.md
├── submission/
│   ├── PRESENTATION.md
│   └── DEMO.md
├── src/
│   ├── __init__.py
│   ├── control_interface.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── hardware_ui.py
│   ├── impulse.py
│   ├── make_impulse.py
│   ├── nlms.py
│   ├── pipeline.py
│   ├── room_sim.py
│   ├── spectral_mask.py
│   └── vad_fast.py
├── data/
│   ├── clean_speech/
│   ├── mixed/
│   └── noise/
├── outputs/
├── build_samples.py
├── demo.py
├── validate.py
├── visualize.py
├── requirements.txt
├── docs/
│   └── architecture.md
└── assets/
    └── screenshots/
```

---

## 15. Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
pip install -r requirements.txt
```

Alternatively, create a virtual environment first:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If the interactive demo is enabled:

```bash
pip install streamlit
```

---

## 16. Run

### Run Validation

```bash
python -m src.validate
```

The validation pipeline evaluates the ANC system across:

```text
-3 dB SNR
 0 dB SNR
+3 dB SNR
```

It reports speech-intelligibility and SNR improvements before and after processing.

### Generate Spectrograms

```bash
python -m src.visualize
```

The visualization generates a three-panel STFT comparison:

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

### Hardware UI Simulator

To launch the GUI simulator representing the physical hardware box enclosure and OLED interface:

```bash
hardware_ui.py  

---

## 17. Results

The current software evaluation pipeline reports the following results:

| Configuration | Input SNR | STOI In | STOI Out | STOI Δ | SNR Gain |
|---|---:|---:|---:|---:|---:|
| Full Pipeline | -3.0 dB | 0.7087 | 0.9298 | +0.2211 | +7.95 dB |
| Full Pipeline | 0.0 dB | 0.7854 | 0.9363 | +0.1509 | +5.69 dB |
| Full Pipeline | +3.0 dB | 0.8525 | 0.9533 | +0.1008 | +3.60 dB |

These results demonstrate improvement in speech intelligibility and signal-to-noise ratio across multiple input-noise conditions.

---

## 18. Evaluation Metrics

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

The highest measured improvement in the current results is:

```text
+7.95 dB
```

---

## 19. TinyML Deployment Targets

The TinyML model is intended to remain small enough for embedded execution.

Target characteristics include:

- **Input features:** 13–20 MFCC coefficients
- **Audio context:** 500 ms
- **Hop size:** 100 ms
- **Model:** lightweight 1D CNN / Depthwise Separable Conv1D
- **Quantization:** INT8
- **Target peak RAM:** < 64 KB
- **Target inference latency:** < 20 ms per 100 ms hop
- **Possible deployment:** TensorFlow Lite for Microcontrollers

The model should run asynchronously so that real-time ANC processing is not interrupted.

---


## 20. Future Scope

### 1. Closed-Loop ANC

Extend V1 feedforward ANC to V2 feedback ANC by adding an internal error microphone and implementing **FxNLMS**.

### 2. Defence-Specific Dataset Expansion

Expand the evaluation dataset with representative high-noise environments relevant to defence applications, while maintaining separate files/scenarios for training and validation.

### 3. Embedded TinyML Deployment

Deploy the trained lightweight classifier directly on the ESP32-S3 using an embedded inference framework such as TensorFlow Lite for Microcontrollers.

### 4. Headset / Protective-Headgear Integration

Replace the V1 speaker demonstration output with an appropriate headphone/earcup audio driver and integrate the microphones and processing electronics into a compact wearable form factor.

### 5. Real-Time Hardware Validation

Validate the complete system using physical microphones, controlled acoustic disturbances, and real-time measurements of latency, attenuation, speech intelligibility, and stability.

### 6. Adaptive Multi-Condition Control

Further improve the controller so that DSP parameters automatically adapt to changing stationary, non-stationary, speech-dominant, and transient acoustic conditions.

---

## 21. Important Notes

- V1 uses **two microphones** and implements feedforward adaptive noise reduction.
- V2 adds a **third internal error microphone** for feedback/closed-loop ANC and FxNLMS.
- The reference microphone should be described as an **Ambient Noise Reference**, not as capturing noise only.
- The V1 MAX98357A is intended for the **speaker demonstration**; it is not the final headphone driver.
- Exact ESP32-S3 GPIO assignments depend on the selected development board and should be finalized from its schematic/pinout.
- Power values in the hardware report are design estimates and should be verified against the exact component datasheets and experimental measurements.
- Do not upload passwords, API keys, access tokens, `.env` files containing secrets, or other confidential credentials to the repository.
