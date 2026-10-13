#include "EQStage.h"
#include <algorithm>
#include <cmath>

namespace Crunchypy
{

namespace
{
constexpr float kKnobThreshold = 1e-4f;
constexpr float kPi = 3.14159265358979323846f;

struct BandSpec
{
    float fc;       // Measured resonant center frequency in Hz
    float Q;        // Measured resonant quality factor
    float gMax;     // Max boost in dB at k = 1.0
    float gMin;     // Max cut in dB at k = 0.0 (positive number)
};

// Calibrated from LiveSPICE schematic analysis and empirical transfer functions:
// Mesa labels: 80, 240, 750, 2200, 6600
constexpr BandSpec kBandSpecs[5] =
{
    {   90.8f, 1.15f, 17.24f, 15.78f }, // Faceplate 80 Hz label
    {  398.4f, 1.53f, 16.31f, 14.98f }, // Faceplate 240 Hz label (physical tank is ~400 Hz)
    {  723.6f, 1.82f, 16.17f, 14.85f }, // Faceplate 750 Hz label
    { 1646.5f, 0.54f, 11.52f, 10.33f }, // Faceplate 2200 Hz label (damped by 1k R14)
    { 4998.0f, 0.84f, 12.17f, 10.86f }  // Faceplate 6600 Hz label (damped by 1k R15)
};
} // namespace

EQStage::EQStage()
{
    knobs.fill (0.5f);
    reset();
    computeCoefficients();
}

bool EQStage::prepare (double sampleRate)
{
    fs = sampleRate > 1000.0 ? sampleRate : 48000.0;
    reset();
    computeCoefficients();
    return true;
}

void EQStage::reset() noexcept
{
    for (auto& s : state)
    {
        s.z1 = 0.0f;
        s.z2 = 0.0f;
    }
}

void EQStage::setKnobs (const std::array<float, 5>& k) noexcept
{
    bool dirty = false;
    for (size_t i = 0; i < 5; ++i)
    {
        const float diff = std::abs (k[i] - knobs[i]);
        if (diff > kKnobThreshold)
        {
            knobs[i] = k[i];
            dirty = true;
        }
    }
    if (dirty)
        computeCoefficients();
}

void EQStage::computeCoefficients() noexcept
{
    // Option 1: Linear dB Slider Law across travel [0, 1]
    // k = 0.5 is exactly 0 dB (noon flat response)
    std::array<float, 5> rawGain {};
    float boostSum = 0.0f;
    float cutSum = 0.0f;

    for (size_t i = 0; i < 5; ++i)
    {
        const float k = std::clamp (knobs[i], 0.0f, 1.0f);
        const float u = 2.0f * (k - 0.5f); // in [-1, +1]

        if (u >= 0.0f)
            rawGain[i] = u * kBandSpecs[i].gMax;
        else
            rawGain[i] = u * kBandSpecs[i].gMin; // u is negative -> rawGain is negative dB

        if (rawGain[i] > 0.0f)
            boostSum += rawGain[i];
        else
            cutSum += -rawGain[i];
    }

    // Circuit multi-band summing junction coupling compression
    const float compBoost = 1.0f / (1.0f + 0.009f * std::max (0.0f, boostSum - 17.0f));
    const float compCut   = 1.0f / (1.0f + 0.009f * std::max (0.0f, cutSum - 16.0f));

    // Design 5 RBJ Peaking EQ Biquads
    const float fSample = static_cast<float> (fs);

    for (size_t i = 0; i < 5; ++i)
    {
        const float g = rawGain[i] * (rawGain[i] >= 0.0f ? compBoost : compCut);

        if (std::abs (g) < 0.001f)
        {
            coeffs[i] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            continue;
        }

        const float A = std::pow (10.0f, g / 40.0f);
        const float w0 = 2.0f * kPi * kBandSpecs[i].fc / fSample;
        const float cosw0 = std::cos (w0);
        const float sinw0 = std::sin (w0);
        const float alpha = sinw0 / (2.0f * kBandSpecs[i].Q);

        const float b0 = 1.0f + alpha * A;
        const float b1 = -2.0f * cosw0;
        const float b2 = 1.0f - alpha * A;
        const float a0 = 1.0f + alpha / A;
        const float a1 = -2.0f * cosw0;
        const float a2 = 1.0f - alpha / A;

        const float invA0 = 1.0f / a0;
        coeffs[i] = { b0 * invA0, b1 * invA0, b2 * invA0, a1 * invA0, a2 * invA0 };
    }
}

float EQStage::process (float x) noexcept
{
    if (! enabled)
        return x;

    // 5-biquad cascade in Direct Form II Transposed
    for (size_t i = 0; i < 5; ++i)
    {
        const auto& c = coeffs[i];
        auto& s = state[i];

        const float y = c.b0 * x + s.z1;
        s.z1 = c.b1 * x - c.a1 * y + s.z2;
        s.z2 = c.b2 * x - c.a2 * y;
        x = y;
    }

    return x;
}

} // namespace Crunchypy