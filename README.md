# CrunchyPy 🎸⚡

**A component-accurate, real-time neural network and physical DSP simulation of the holy grail Mesa/Boogie® Mark IIC+ (featuring the coveted "++" circuit mod).**

Built with modern C++, [JUCE](https://github.com/juce-framework/JUCE), and [RTNeural](https://github.com/Yahiake/RTNeural), CrunchyPy delivers the raw, biting, articulate high-gain punch that defined legendary records like Metallica's *Master of Puppets* and Dream Theater's *Images and Words* — with zero compromise on latency or CPU performance.

Available as **VST3** and **Standalone** on **Windows** and **Linux** (macOS compatible).

---

## ⚡ What Makes CrunchyPy Different?

Most neural amp sims treat the entire amplifier as a black box: you train one model on audio in and audio out, and hope that moving virtual knobs doesn't sound smeared or unnatural. 

Real tube amplifiers don't work like that — and the Mark IIC+ definitely doesn't. 

In the real Mark IIC+, the signal flows through an intricate cascade:
1. An aggressive, sensitive preamp with interactive Pull switches and tube saturation stages.
2. A discrete passive **5-band Graphic EQ** using physical inductors and capacitors (LC tanks) wired to a shared summing junction.
3. The **Master 1** inter-stage volume control driving the power amp.
4. A high-headroom **100W Simul-Class / 6L6 power section** with negative feedback presence shaping.

CrunchyPy reflects this exact physical topology. Every stage is modeled independently with dedicated DSP and neural networks:

```
[ Guitar DI ]
      │
      ▼
┌──────────────┐
│  Input Trim  │ (±24 dB audio interface calibration)
└──────┬───────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Preamp Neural Network (GRU, 13 features, 96 hidden)   │
│  • Volume 1 (Pull Bright)    • Treble (Pull Shift)     │
│  • Bass (Pull Shift)         • Middle (Pull ++ Mod)    │
│  • Lead Drive                • Lead Master (Pull Bright)│
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│  Graphic EQ (Direct Form II Transposed LC Tank Array)  │
│  • 80 Hz (90.8 Hz)           • 240 Hz (398.4 Hz)       │
│  • 750 Hz (723.6 Hz)         • 2200 Hz (1646.5 Hz)     │
│  • 6600 Hz (4998.0 Hz)       • Coupling Compression    │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│  Master 1 Voltage Divide & Pull Deep (Low-End Punch)   │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│  100W Poweramp Neural Network (GRU, 32 hidden units)   │
│  • Presence (Negative Feedback Tube Resonance)         │
│  • +1.14 dB Nominal Level Compensation                 │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│  Master Output Trim & Transparent Soft-Knee Limiter    │
└────────────────────────────────────────────────────────┘
```

---

## 🎛 The Circuit & Controls

### 1. Preamp Stage (Neural Recurrent Network)
Trained on thousands of circuit state iterations, the preamp uses a custom recurrent Gated Recurrent Unit (GRU) with 96 hidden units. It captures non-linear tube conduction, dynamic compression, and inter-stage clipping across all 12 controls:

* **VOLUME 1**: Sets the input gain into the first 12AX7 stages. High settings saturate the front-end before the tone stack.
* **PULL BRIGHT (Vol 1)**: Engages the bright bypass capacitor across the volume pot, preserving crystalline high-end at lower volume settings.
* **TREBLE**: The primary driver of the Mark series tone stack. In a Mark IIC+, treble sits early in the circuit and dictates overall gain as well as frequency response.
* **PULL SHIFT (Treble)**: Shifts the tone stack turnover frequency for a more aggressive upper-mid bite.
* **BASS & PULL SHIFT (Bass)**: Controls low-end response before distortion occurs. Keep this modest (2–3) for tight metal rhythm, and pull shift for deep sub-thump.
* **MIDDLE**: Mid-range body.
* **PULL ++ (Middle)**: Engages the fabled **"++" factory modification** — re-voicing the intermediate gain stages for a tighter, hotter, and more saturated lead response.
* **LEAD DRIVE**: Controls the gain of the cascading lead drive tube stages.
* **LEAD MASTER & PULL BRIGHT**: Controls lead channel level with an optional high-frequency edge boost.
* **PREAMP IN Toggle**: Seamlessly bypasses the preamp stage if you wish to run external preamps into the Mark EQ and poweramp.

### 2. Physical LC Tank 5-Band Graphic EQ
The Mesa 5-band graphic EQ is famous for a reason — and it is **not** a standard digital peaking EQ. In the physical chassis, five series LC traps (`L1..L5`, `C5..C9`, `R11..R15`) shunt frequencies into an active summing junction. 

Rather than using an idealized textbook filter, CrunchyPy's filters were calibrated directly against LiveSPICE circuit measurements of the physical schematic:

| Faceplate Label | Physical Resonance | Measured Peak / Notch | Max Range | Filter Q |
| :--- | :--- | :--- | :--- | :--- |
| **80 Hz** | `L1 = 1.0 H, C5 = 3.0 µF` | **90.8 Hz** | +17.2 dB / -15.8 dB | 1.15 |
| **240 Hz** | `L2 = 330 mH, C6 = 470 nF` | **398.4 Hz** *(Real circuit shift)* | +16.3 dB / -15.0 dB | 1.53 |
| **750 Hz** | `L3 = 220 mH, C7 = 220 nF` | **723.6 Hz** | +16.2 dB / -14.9 dB | 1.82 |
| **2200 Hz** | `L4 = 68 mH, C8 = 150 nF` | **1646.5 Hz** *(R14 damped)* | +11.5 dB / -10.3 dB | 0.54 |
| **6600 Hz** | `L5 = 33 mH, C9 = 33 nF` | **4998.0 Hz** *(R15 damped)* | +12.2 dB / -10.9 dB | 0.84 |

* **Coupling Compression**: When all five sliders are pushed, physical transistor interaction naturally compresses the combined boost, capping the accumulation at **+21.6 dB** rather than runaway digital clipping.
* **Direct Form II Transposed**: Built with instantaneous block-rate parameter slew ($\tau = 15\text{ ms}$) so slider adjustments respond immediately with zero zipper noise or multi-second lag.
* **EQ IN Toggle**: Independent clean bypass switch.

### 3. Master 1 & 100W Poweramp Stage
* **MASTER 1 & PULL DEEP**: Inter-stage voltage divider controlling the drive level into the graphic EQ and power amplifier. Pull Deep kicks in heavy cabinet low-end resonance.
* **100W 6L6 Poweramp (GRU Network)**: A 32-hidden-unit neural network reproducing the sag, punch, and compression of four 6L6 power tubes driving an output transformer.
* **PRESENCE**: Negative feedback frequency control in the power amp stage. Controls sizzle and bite directly in the poweramp neural net.
* **Volume Alignment**: Bypassing the poweramp (`PWR IN` switch) features automatic **+1.14 dB** nominal compensation, so you can test raw preamp signals against full amp tones without startling volume jumps.

### 4. Utility & Safety Section
* **INPUT & OUTPUT TRIMS**: Dedicated input gain ($\pm 24\text{ dB}$) to calibrate your audio interface's instrument DI level, and master output level ($-24\text{ to }+6\text{ dB}$).
* **LIMITER TOGGLE**: Transparent soft-knee safety limiter at the end of the chain. Unlike aggressive master limiters, this limiter stays at strictly **1.000000× unity gain** below threshold ($-0.92\text{ dBFS}$) with zero small-signal pumping or coloration. Default is OFF.
* **MONO / STEREO**: Starts in **MONO** mode (channel 0 processing, duplicated to outputs for optimal single-guitar CPU performance). Clicking toggles to **STEREO** for true dual-mono stereo tracks or stereo chorus/delays placed ahead of the amp.

---

## 🎧 Dialing in the Classic Tones

If you have never dialed in a real Mesa Mark series amp before, the tone stack can surprise you: **the controls are highly interactive!**

### 1. The Classic "V-Curve" Thrash Rhythm (*Master of Puppets* / *Justice*)
* **Volume 1**: `7.0 – 8.0` (Pushed hard)
* **Treble**: `7.5` (Most of the distortion comes from Treble in this circuit!)
* **Bass**: `2.0` (Keep it low! High bass in the preamp gets muddy. Add bass back with the Graphic EQ!)
* **Middle**: `3.5`
* **Lead Drive**: `8.0`
* **Lead Master**: `3.5`
* **Master 1**: `4.5`
* **Presence**: `3.0`
* **Graphic EQ**: Set the classic **"V"**:
  * `80 Hz`: Boosted to ~`80%`
  * `240 Hz`: Around `65%`
  * `750 Hz`: Scooped down to `15% – 20%`
  * `2200 Hz`: Around `65%`
  * `6600 Hz`: Boosted to ~`80%`

### 2. Singing Fusion Lead (Santana / Petrucci Solo Tone)
* **Volume 1**: `6.5`
* **Lead Drive**: `6.5`
* **Lead Master**: `4.5`
* **Treble**: `6.0`
* **Bass**: `3.5`
* **Middle**: `5.5`
* **Presence**: `2.5` (Smooth off the top end)
* **Graphic EQ**: Smooth smile curve; boost `750 Hz` slightly for vocal mid-range feedback.

---

## 🛠 Building from Source

### Prerequisites
* **CMake** (v3.22 or higher)
* A modern C++ compiler supporting C++17:
  * **Windows**: Visual Studio 2022 (MSVC v143+)
  * **Linux**: GCC 10+ or Clang 12+
* **Git** (with recursive submodule support)

### 1. Clone with Submodules
```bash
git clone --recursive https://github.com/Yahiake/CrunchyPy.git
cd CrunchyPy
```
*(If you already cloned without submodules, run: `git submodule update --init --recursive`)*

### 2. Linux Dependencies (Ubuntu / Debian)
```bash
sudo apt-get update && sudo apt-get install -y \
  pkg-config \
  libasound2-dev \
  libjack-jackd2-dev \
  libfreetype6-dev \
  libfontconfig1-dev \
  libgl1-mesa-dev \
  libx11-dev \
  libxinerama-dev \
  libxext-dev \
  libxcursor-dev \
  libxrandr-dev \
  libxrender-dev \
  libxcomposite-dev
```

### 3. Build (Windows & Linux)
```bash
# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build build --config Release --parallel
```

On Windows, the packaged binaries will automatically be staged in the `dist/` folder:
* `CrunchyPy.vst3` (VST3 plugin bundle)
* `CrunchyPy.exe` (Standalone application)

On Linux:
* VST3: `build/CrunchyPy_artefacts/VST3/CrunchyPy.vst3`
* Standalone: `build/CrunchyPy_artefacts/Standalone/CrunchyPy`

---

## 🚀 Performance & Real-Time Safety

* **Zero Latency**: Native 48 kHz operation has strictly **0 samples of latency**. 
* **Bandlimited Resampling**: If your host runs at 44.1 kHz, 96 kHz, or 192 kHz, CrunchyPy transparently employs a low-latency Kaiser-sinc resampler ($M=24$) so the neural networks and physical filter tanks always execute on their native 48 kHz clock.
* **SIMD Accelerated**: Powered by [RTNeural](https://github.com/Yahiake/RTNeural) with **xsimd** AVX2 vectorization, running full 256-bit SIMD matrix-vector kernels with minimal CPU utilization.
* **Denormal Safe**: Hardware-level FTZ (Flush-To-Zero) and DAZ (Denormals-Are-Zero) protection throughout the entire signal chain.

---

## 📜 License & Credits

* **Developer**: Yahia Kemari ([@Yahiake](https://github.com/Yahiake))
* **Inference Engine**: [RTNeural](https://github.com/Yahiake/RTNeural) by Jatin Chowdhury ([@jatinchowdhury18](https://github.com/jatinchowdhury18))
* **Framework**: [JUCE 8](https://github.com/juce-framework/JUCE)
* **License**: Released under the [GNU General Public License v3.0](LICENSE).

*Disclaimer: Mesa/Boogie, Mark IIC+, and Mark IIC++ are registered trademarks of Gibson Brands, Inc. / Mesa/Boogie Ltd. CrunchyPy is an independent open-source research and engineering project not affiliated with, endorsed, or sponsored by Mesa/Boogie or Gibson Brands.*
