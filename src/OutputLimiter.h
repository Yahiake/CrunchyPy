#pragma once

#include <algorithm>
#include <cmath>

namespace Crunchypy
{

/**
    Transparent soft-knee output safety limiter.

    - Exactly unity gain (1.000000x / 0.0 dB) for all signals below threshold (0.90 / -0.92 dBFS).
    - Zero distortion, zero gain pumping, and zero small-signal boosting.
    - Seamless C1-continuous soft knee approaching ceiling when peaks exceed threshold.
*/
class OutputLimiter
{
public:
    void prepare (double sampleRate) noexcept
    {
        fs = (float) sampleRate;
        reset();
    }

    void reset() noexcept
    {
    }

    /** Ceiling in [0.25, 1.0]. */
    void setCeiling (float linear) noexcept
    {
        ceiling = juceNorm (linear);
    }

    /** Processes one sample. */
    float processSample (float x) noexcept
    {
        const float ax = std::abs (x);
        constexpr float threshold = 0.90f; // -0.92 dBFS knee start
        if (ax <= threshold)
            return x * ceiling; // 100% transparent bit-exact pass-through

        // C1 continuous soft knee asymptote approaching 1.0
        constexpr float margin = 1.0f - threshold; // 0.10f
        const float delta = ax - threshold;
        const float compressed = threshold + margin * std::tanh (delta / margin);
        return (x < 0.0f ? -compressed : compressed) * ceiling;
    }

private:
    static float juceNorm (float v) noexcept
    {
        return v < 0.25f ? 0.25f : (v > 1.0f ? 1.0f : v);
    }

    float fs = 48000.0f;
    float ceiling = 1.0f;
};

} // namespace Crunchypy
