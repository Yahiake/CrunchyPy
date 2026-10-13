# CrunchyPy 🎸⚡
### A Neural & Physical Tribute to the Mesa/Boogie® Mark IIC+ (and the Mythical "++" Mod)

> *"Man, that little thing really boogies!"*  
> — Carlos Santana, playing Randall Smith's hot-rodded Princeton in 1969.

Built with modern C++, [JUCE](https://github.com/juce-framework/JUCE), and [RTNeural](https://github.com/Yahiake/RTNeural), **CrunchyPy** is an open-source, component-accurate real-time simulation of the holy grail of high-gain guitar amplification: the **Mesa/Boogie® Mark IIC+**, complete with the factory **"++" hot-rod gain mod**.

Available as **VST3** and **Standalone** on **Windows** and **Linux** (macOS compatible).

---

## 📖 The Legend: Why the Mark IIC+ is the Holy Grail

In the pantheon of guitar tone, there are legendary amplifiers, and then there is the **Mark IIC+**.

Manufactured in Petaluma, California for only about twelve months between 1984 and 1985 (roughly serial numbers 11,000 to 14,400, with fewer than 3,000 units ever leaving the shop), the Mark IIC+ was born when **Randall Smith** and master technician **Mike Bendinelli** reworked the earlier IIC lead circuit. They overhauled the lead drive cascade, altered the power supply response, and unlocked a ferocious, articulate, singing gain structure unlike anything the world had heard.

When an amp left the factory with the updated circuit—or when an owner shipped their chassis back to Petaluma to be upgraded by Mike Bendinelli himself—Mike marked a black Sharpie **`+`** directly above the power cord socket on the rear panel. That simple handwritten plus sign became the most sought-after mark in guitar history.

### The Mythical "++" Mod
For players who demanded even more searing front-end saturation, tighter low-end filtering, and an extra surge of harmonic teeth, Mike Bendinelli performed custom factory modifications known as the **"++" mod**. An extra boost stage was grafted into the intermediate gain cascade, turning the already-legendary lead channel into an unrelenting, hyper-focused flamethrower. Mike would mark a second plus sign in Sharpie: **`++`**. 

### The "Crunch Berries" Head & The Sound of Thrash
In autumn 1985, **James Hetfield** and **Kirk Hammett** walked into Sweet Silence Studios in Copenhagen with producer Flemming Rasmussen to track Metallica's *Master of Puppets*. James brought along an original Mesa Mark IIC+ head adorned with a Cap'n Crunch **"Crunch Berries"** cereal sticker across the front faceplate (alongside another head nicknamed "Emma"). 

By starving the preamp of low-end mud, pushing the front-end tubes hard, and carving an extreme "V-curve" into the onboard 5-band graphic EQ, James dialed in the most influential metal rhythm tone in recorded history: dry, percussive, punishing, and surgical enough to track quadrupled downpicked riffs at 220 BPM with razor-wire clarity. 

*(This project was born out of an obsession with that exact sound—hence the codename **ProjectCrunchyBerries**).*

From the crushing brutality of *Master of Puppets* and *...And Justice for All*, to **John Petrucci's** liquid, singing flute-like leads on Dream Theater's *Images and Words* and *Awake*, to **Prince's** blistering solos and **Steve Lukather's** pristine session work—the Mark IIC+ defined the golden era of guitar tone.

---

## ⚡ Architecture: Why Component Separation Matters

Most neural amp plugins treat the amplifier as a single "black box": you feed audio in, take audio out, and train one neural net on everything.

**That approach fails completely on a Mark IIC+.**

On a Mark series amplifier, the interaction between the pre-gain tone stack, the cascading lead tube stages, the passive LC-tank Graphic EQ, and the poweramp's negative feedback presence circuit is deeply non-linear and modular. If you don't model them in their physical circuit positions, the controls smear together and lose their magic.

CrunchyPy mirrors the exact physical signal topology of the amplifier:

```
                      [ Raw Instrument DI ]
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Input Gain Utility    │ (±24 dB audio interface trim)
                   └────────────┬────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────────┐
     │  1. Preamp Neural Network (Recurrent GRU, 96 Hidden)    │
     │  • Volume 1 (Pull Bright)      • Treble (Pull Shift)    │
     │  • Bass (Pull Shift)           • Middle                 │
     │  • Pull ++ (Mythical Gain Mod) • Lead Drive             │
     │  • Lead Master (Pull Bright)   • PREAMP IN Bypass       │
     └──────────────────────────┬──────────────────────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────────┐
     │  2. 5-Band Graphic EQ (Physical LC Tank Biquad Array)   │
     │  • 80 Hz (90.8 Hz)             • 240 Hz (398.4 Hz)      │
     │  • 750 Hz (723.6 Hz)           • 2200 Hz (1646.5 Hz)    │
     │  • 6600 Hz (4998.0 Hz)         • Summing Compression    │
     │  • Direct Form II Transposed   • EQ IN Bypass           │
     └──────────────────────────┬──────────────────────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────────┐
     │  3. Master 1 Voltage Divide & Pull Deep Resonance       │
     └──────────────────────────┬──────────────────────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────────┐
     │  4. 100W Poweramp Neural Network (Recurrent GRU, 32 H)  │
     │  • Negative Feedback Presence Control                   │
     │  • +1.14 dB Nominal Level Compensation / PWR IN Bypass  │
     └──────────────────────────┬──────────────────────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────────┐
     │  5. Master Output Trim & Transparent Soft-Knee Limiter  │
     └─────────────────────────────────────────────────────────┘
```

---

## 🎛 The Circuit & Controls in Detail

### 1. The Preamp: Pre-Gain Tone Stack & Tube Cascade
Unlike a Marshall JCM800 or Peavey 5150 where the tone stack sits *after* the preamp distortion stages, the Mark series tone stack sits **early in the circuit** (right after V1):

* **VOLUME 1**: Sets the input sensitivity and pushes the first 12AX7 tube stage into edge-of-breakup harmonic warmth.
* **PULL BRIGHT (Vol 1)**: Engages a 120 pF bright bypass capacitor across the volume pot, giving glassy sparkle at lower volume settings.
* **TREBLE**: The most powerful knob on the amp. Because Treble sits before the lead drive stages, **it acts as a primary gain and saturation control**. Pushing Treble feeds more raw signal into the lead tubes.
* **PULL SHIFT (Treble)**: Shifts the tone stack turnover frequency for a more aggressive upper-mid throatiness.
* **BASS & PULL SHIFT (Bass)**: Controls low-end before clipping. **Golden Rule: Keep Bass low (1.5 – 3.0)!** If you crank Bass in the preamp, the tubes will choke on low-frequency mud and fart out. The thunderous low end is added back *after* distortion using the Graphic EQ! Pull Shift changes the bass slope resonance.
* **MIDDLE**: Controls midrange body and thickness.
* **PULL ++ (Middle)**: Engages the fabled **Mike Bendinelli "++" factory modification**—re-voicing the intermediate gain stages for a blistering surge in saturation, pick attack, and tight harmonic bite.
* **LEAD DRIVE**: Dial in the cascading high-gain lead stages. Pushed past 7, it transitions into thick, saturated, harmonic sustain.
* **LEAD MASTER & PULL BRIGHT**: Controls the lead channel output level into Master 1, with an optional top-end pull bite.
* **PREAMP IN**: Dedicated toggle switch to bypass the neural preamp (perfect if you want to feed your own hardware preamp pedals or external models into the Mark EQ and poweramp).

---

### 2. The 5-Band Graphic EQ: Physical LC Tank Filters
The Mesa 5-band Graphic EQ is legendary—and it is **not** a standard digital bell filter. In the physical amplifier, five passive series inductors and capacitors (LC resonant traps) return to a shared summing junction (`L1..L5`, `C5..C9`, `R11..R15`).

CrunchyPy's EQ filters were calibrated directly against LiveSPICE circuit simulations of the physical schematic:

| Faceplate Label | Physical Tank Components | Measured Center Frequency | Max Range | Filter Q |
| :---: | :---: | :---: | :---: | :---: |
| **80 Hz** | `L1 = 1.0 H,  C5 = 3.0 µF` | **90.8 Hz** | +17.2 dB / -15.8 dB | 1.15 |
| **240 Hz** | `L2 = 330 mH, C6 = 470 nF` | **398.4 Hz** *(Real circuit shift)* | +16.3 dB / -15.0 dB | 1.53 |
| **750 Hz** | `L3 = 220 mH, C7 = 220 nF` | **723.6 Hz** | +16.2 dB / -14.9 dB | 1.82 |
| **2200 Hz** | `L4 = 68 mH,  C8 = 150 nF` | **1646.5 Hz** *(Damped by R14)* | +11.5 dB / -10.3 dB | 0.54 |
| **6600 Hz** | `L5 = 33 mH,  C9 = 33 nF`  | **4998.0 Hz** *(Damped by R15)* | +12.2 dB / -10.9 dB | 0.84 |

* **Summing Junction Compression**: When all 5 sliders are boosted, physical transistor interaction naturally compresses the total gain, enforcing a strict **+21.6 dB ceiling** just like the real chassis.
* **Direct Form II Transposed**: Built in DF2T with instantaneous block-rate parameter smoothing ($\tau = 15\text{ ms}$) so slider adjustments respond immediately under your mouse with zero zipper noise or multi-second lag.
* **EQ IN**: Clean bypass toggle switch.

---

### 3. Master 1 & 100W Poweramp Section
* **MASTER 1**: The physical voltage divider sitting between the preamp/EQ and the poweramp, dictating how hard the power tubes are driven.
* **PULL DEEP (Master 1)**: Pull Deep engages deep cabinet low-end resonance, adding punch and thud to the entire rig.
* **100W Poweramp (Neural GRU)**: A 32-hidden-unit neural recurrent model capturing the dynamic sag, transformer saturation, and compression of four 6L6 tubes pushing high-current iron.
* **PRESENCE**: Negative feedback frequency control in the poweramp stage. Rather than an EQ filter, Presence modulates the power section's feedback loop directly in the neural net.
* **Level-Aligned Bypass (PWR IN)**: Features automatic **+1.14 dB** nominal volume compensation, so toggling between raw preamp signal and full poweramp never blasts your ears or monitor levels.

---

### 4. Utility & Safety Section
* **INPUT TRIM (±24 dB)**: Calibrate your audio interface's instrument input level to hit the virtual preamp tubes at their exact sweet spot.
* **OUTPUT TRIM (-24 to +6 dB)**: Master output volume trim.
* **LIMITER TOGGLE**: Transparent soft-knee safety limiter at the end of the chain. Unlike brickwall limiters that squash guitar dynamics, this limiter stays at strictly **1.000000× unity gain** below threshold ($-0.92\text{ dBFS}$) with zero small-signal pumping or coloration. Default is OFF.
* **MONO / STEREO**: Defaults to **MONO** (channel 0 processing duplicated to outputs, saving 50% CPU for standard guitar DI tracking). Click to switch to **STEREO** for true dual-mono stereo tracks or stereo chorus/delays placed in front of the amp.

---

## 🎸 The Dialing Bible: How to Get The Tones

Dialing in a Mark IIC+ is fundamentally different from dialing in a Marshall or 5150. **Remember: Treble adds gain, Bass adds mud, and the Graphic EQ shapes the world.**

### 1. James Hetfield's "Crunch Berries" *Master of Puppets* Thrash Tone
*Tight, dry, punishing, and surgical. Zero flub on high-speed palm mutes.*
* **Volume 1**: `8.0 – 10.0` (Pushed to the max)
* **Treble**: `7.5`
* **Pull Shift (Treble)**: **ON**
* **Bass**: `2.0` *(Do not crank this! Keep it at 2.0!)*
* **Middle**: `3.5`
* **Pull ++ (Middle)**: **ON** (Extra saturation and bite)
* **Lead Drive**: `8.0`
* **Lead Master**: `7.0`
* **Master 1**: `5.0`
* **Pull Deep (Master 1)**: **ON**
* **Presence**: `4.5`
* **5-Band Graphic EQ ("The Classic V")**:
  * `80 Hz`: Boosted to **~80%**
  * `240 Hz`: Around **50% – 65%**
  * `750 Hz`: Scooped down to **0% – 10%** *(The secret to the thrash scoop!)*
  * `2200 Hz`: Around **50% – 65%**
  * `6600 Hz`: Boosted to **~80%**

### 2. John Petrucci's *Images and Words* / *Awake* Singing Lead
*Liquid, flute-like sustain that holds notes for days without turning harsh.*
* **Volume 1**: `6.5`
* **Lead Drive**: `6.5`
* **Lead Master**: `4.5`
* **Treble**: `6.0`
* **Bass**: `3.5`
* **Middle**: `5.5`
* **Presence**: `2.5` *(Rolled back for smooth, vocal warmth)*
* **Master 1**: `5.5`
* **Graphic EQ**: Slight smile curve; keep `750 Hz` near `40% – 50%` so the midrange cuts through the mix like a laser.

### 3. '80s Studio Clean & Edge-of-Breakup (Lukather / Prince)
*Turn the Lead channel off, push the poweramp, and pull the bright switches.*
* **Volume 1**: `4.5`
* **Pull Bright (Vol 1)**: **ON**
* **Treble**: `6.5`
* **Bass**: `4.5`
* **Middle**: `5.0`
* **Lead Drive**: `0.0` (Lead channel off)
* **Master 1**: `6.0 – 8.0`
* **Presence**: `5.0`
* **Graphic EQ**: Flat or subtle high-frequency chime boost at 6600 Hz.

---

## 🛠 Building from Source

### Prerequisites
* **CMake** (v3.22 or higher)
* A modern C++ compiler supporting C++17 (MSVC v143+ on Windows, GCC 10+ or Clang 12+ on Linux)
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

On Windows, the packaged binaries will automatically be staged into `dist/`:
* `CrunchyPy.vst3` (VST3 plugin bundle)
* `CrunchyPy.exe` (Standalone application)

On Linux:
* VST3: `build/CrunchyPy_artefacts/VST3/CrunchyPy.vst3`
* Standalone: `build/CrunchyPy_artefacts/Standalone/CrunchyPy`

---

## 🚀 Performance & Audio Engine Specs

* **Zero Latency**: Native 48 kHz operation has strictly **0 samples of latency**.
* **Bandlimited Resampling**: Running at 44.1 kHz, 96 kHz, or 192 kHz? CrunchyPy transparently employs a low-latency Kaiser-sinc resampler ($M=24$) so the neural networks and physical filter tanks always execute on their native 48 kHz clock.
* **SIMD Vectorization**: Powered by [RTNeural](https://github.com/Yahiake/RTNeural) with **xsimd** AVX2 vectorization, running full 256-bit SIMD matrix-vector kernels with minimal CPU utilization.
* **Denormal Protection**: Hardware-level FTZ (Flush-To-Zero) and DAZ (Denormals-Are-Zero) protection throughout the entire signal chain.

---

## 📜 Credits & Acknowledgments

* **Architecture, Preamp Circuit & Lead Developer**:  
  **Yahia Kemari** ([@Yahiake](https://github.com/Yahiake))  
  *Schematic capture of the Mark IIC+(++) preamp section, neural network training pipeline, plugin architecture, DSP engine, and front-panel UI design.*

* **Graphic EQ & Poweramp Circuit Schematics**:  
  Special thanks to **Emy** ([@MARKTHERENCE](https://github.com/MARKTHERENCE)) for crafting the detailed LiveSPICE circuit schematics for the 5-band Graphic EQ and 100W Simul-Class/6L6 Poweramp section.  
  Check out their circuit repository: [MARKTHERENCE/LiveSPICE-MK-Circuits](https://github.com/MARKTHERENCE/LiveSPICE-MK-Circuits/tree/main/amps).

* **Inference Engine**:  
  [RTNeural](https://github.com/Yahiake/RTNeural) by **Jatin Chowdhury** ([@jatinchowdhury18](https://github.com/jatinchowdhury18)).

* **Audio Framework**:  
  [JUCE 8](https://github.com/juce-framework/JUCE).

* **License**:  
  Released under the [GNU General Public License v3.0](LICENSE).

---

*Disclaimer: Mesa/Boogie, Mark IIC+, Mark IIC++, and Simul-Class are registered trademarks of Gibson Brands, Inc. / Mesa/Boogie Ltd. CrunchyPy is an independent open-source research and engineering tribute and is not affiliated with, endorsed, or sponsored by Mesa/Boogie or Gibson Brands.*