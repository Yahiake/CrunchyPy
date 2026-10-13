#pragma once
#include <array>
#include <cmath>

namespace Crunchypy
{

/**
    Calibrated 5-Band Graphic EQ stage matching the Mesa Boogie Mark IIC+ circuit.
    
    Architecture:
      - 5 cascaded peaking biquads in Direct Form II Transposed.
      - Resonant center frequencies and Q factors calibrated directly to the
        circuit's physical LC tanks (L1..L5, C5..C9, R11..R15):
          Faceplate 80 Hz   -> 90.8 Hz  (Q = 1.15, +17.2 dB boost / -15.8 dB cut)
          Faceplate 240 Hz  -> 398.4 Hz (Q = 1.53, +16.3 dB boost / -15.0 dB cut)
          Faceplate 750 Hz  -> 723.6 Hz (Q = 1.82, +16.2 dB boost / -14.9 dB cut)
          Faceplate 2200 Hz -> 1646.5 Hz(Q = 0.54, +11.5 dB boost / -10.3 dB cut)
          Faceplate 6600 Hz -> 4998.0 Hz(Q = 0.84, +12.2 dB boost / -10.9 dB cut)
      - Option 1 Linear dB Taper:
          Provides smooth, uniform dB response across slider travel.
          At slider = 0.5 (center), gain is 0 dB (perfect flat response).
      - Multi-band summing junction coupling compression:
          Limits multi-band accumulation to the circuit's +21.6 dB ceiling.
      - Clean EQ bypass switch.
*/
class EQStage
{
public:
    EQStage();

    /** Prepares the EQ stage for a given sample rate (default 48000 Hz). */
    bool prepare (double sampleRate = 48000.0);

    /** True when the EQ stage is ready for processing. */
    bool isLoaded() const noexcept { return true; }

    /** Zeros the filter delay lines. */
    void reset() noexcept;

    /** Feed the five smoothed knob values in [0, 1]. */
    void setKnobs (const std::array<float, 5>& k) noexcept;

    /** Enable or disable the Graphic EQ stage (bypass switch). */
    void setEnabled (bool active) noexcept { enabled = active; }
    bool isEnabled() const noexcept { return enabled; }

    /** Applies the filter to one sample. */
    float process (float x) noexcept;

private:
    void computeCoefficients() noexcept;

    struct BiquadCoeffs
    {
        float b0 = 1.0f;
        float b1 = 0.0f;
        float b2 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
    };

    struct BiquadState
    {
        float z1 = 0.0f;
        float z2 = 0.0f;
    };

    double fs = 48000.0;
    bool enabled = true;

    std::array<float, 5> knobs {};
    std::array<BiquadCoeffs, 5> coeffs {};
    std::array<BiquadState, 5> state {};
};

} // namespace Crunchypy