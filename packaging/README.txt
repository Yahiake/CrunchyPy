CRUNCHYPY
==========
A neural emulation of a Mark IIC+ amplifier preamp (lead channel).

What this is
------------
CrunchyPy is not a physical circuit. It is a neural network that has been
trained to imitate the behaviour of a LiveSPICE simulation of a Mark IIC+
preamp, across every knob and pull switch at once.

The ground truth is simulation, not hardware. The reference is the SPICE model,
so when this plugin and the circuit model disagree, the model is right. No
physical amplifier was measured for this project.

What it is accurate to
----------------------
The network reproduces the LiveSPICE model of the preamp within a measured
error, in the region of knob and signal space covered by the training data.
Outside that region it extrapolates, and extrapolated behaviour is not
guaranteed to be musical.

Controls
--------
VOL1          Channel-1 input volume.       Click to pull BRIGHT.
TREBLE BASS   Passive tone stack.           Click to pull SHIFT.
MIDDLE                                        Click to pull SHIFT (++).
MASTER 1      Output voltage divide. 50% is the render reference (unity, the
              default). Click to pull DEEP.
LEAD DRIVE    Lead-channel drive.            No pull.
LEAD MASTER   Lead-channel volume.           Click to pull BRIGHT.
OUTPUT        Post-network trim, -24 dB to +6 dB (0 dB default).

Clicking a knob's body toggles its pull (an amber ring lights while pulled);
dragging up/down sets the value; double-clicking returns a knob to its default.
Bypassing is transparent: the plugin passes audio through unmodified.

Requirements
------------
Windows (x64), Linux (x64), or macOS (Apple Silicon & Intel).
VST3, AU (macOS), or Standalone application. No installation step,
no system drivers, no network access at runtime.

Licence
-------
Released under the GNU General Public License v3.0, which covers the
JUCE framework this plugin is built on. See LICENSE for the full text.