#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace Crunchypy
{

/**
    Low-latency band-limited resampler (Kaiser-windowed sinc, streaming, mono).

    Output sample n sits at input time n * inRate / outRate. An output is emitted as
    soon as the kernel's right half has arrived, so the algorithmic delay is just the
    kernel half-width (getLookahead() input samples), with no block/FFT latency.
    Nothing allocates after prepare().
*/
class KaiserSincResampler
{
public:
    /** Kernel half-width, in samples of the LOWER of the two rates. Quality vs latency. */
    static constexpr int kHalf = 24;
    static constexpr int kTableRes = 1024;

    void prepare (double inRate, double outRate, int maxIn, double cutoffFraction)
    {
        step = inRate / outRate;
        scale = std::min (1.0, outRate / inRate);
        hs = (int) std::ceil ((double) kHalf / scale);

        buildTable (cutoffFraction);

        buf.assign ((size_t) (maxIn + 4 * hs + 16), 0.0f);
        reset();
    }

    void reset()
    {
        std::fill (buf.begin(), buf.end(), 0.0f);
        len = hs;
        pos = (double) hs;
    }

    int getLookahead() const noexcept { return hs; }

    /** Appends `n` input samples (n <= maxIn) and writes up to `maxOut` outputs. */
    int process (const float* in, int n, float* out, int maxOut) noexcept
    {
        std::memcpy (buf.data() + len, in, (size_t) n * sizeof (float));
        len += n;

        const float* t = table.data();
        const float maxX = (float) (table.size() - 2);
        int produced = 0;

        while (produced < maxOut)
        {
            const int i0 = (int) pos;
            if (i0 + hs >= len)
                break;

            const double f = pos - (double) i0;
            const float* x = buf.data() + (i0 - hs + 1);

            // u = scale * (f - k) for k = -hs+1 ... hs, stepping down by `scale`
            float u = (float) (scale * (f + (double) (hs - 1)));
            const float us = (float) scale;
            float acc = 0.0f;

            const int phIdx = (int) (f * (double) kTableRes) % kTableRes;
            for (int j = 0; j < 2 * hs; ++j, u -= us)
            {
                const float xi = (u + (float) kHalf) * (float) kTableRes;
                if (xi <= 0.0f || xi >= maxX)
                    continue;

                const int ix = (int) xi;
                const float fr = xi - (float) ix;
                acc += x[j] * (t[ix] + (t[ix + 1] - t[ix]) * fr);
            }

            acc *= phaseNorm[(size_t) phIdx];
            out[produced++] = acc * (float) scale;
            pos += step;
        }

        // Drop samples no future output can need.
        const int drop = (int) pos - hs;
        if (drop > 0)
        {
            std::memmove (buf.data(), buf.data() + drop, (size_t) (len - drop) * sizeof (float));
            len -= drop;
            pos -= (double) drop;
        }

        return produced;
    }

private:
    static double besselI0 (double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        const double q = x * x * 0.25;
        for (int k = 1; k < 60; ++k)
        {
            term *= q / ((double) k * (double) k);
            sum += term;
            if (term < 1e-14 * sum)
                break;
        }
        return sum;
    }

    void buildTable (double c)
    {
        constexpr double pi = 3.14159265358979323846;
        constexpr double beta = 8.0;   // ~ 80+ dB stopband
        const int size = 2 * kHalf * kTableRes + 2;
        table.assign ((size_t) size, 0.0f);
        const double i0b = besselI0 (beta);

        for (int i = 0; i < size - 1; ++i)
        {
            const double u = (double) i / (double) kTableRes - (double) kHalf;
            const double r = u / (double) kHalf;
            if (std::abs (r) >= 1.0)
                continue;

            const double w = besselI0 (beta * std::sqrt (1.0 - r * r)) / i0b;
            const double a = c * u;
            const double s = std::abs (a) < 1e-12 ? 1.0 : std::sin (pi * a) / (pi * a);
            table[(size_t) i] = (float) (c * s * w);
        }

        // Per-subphase normalization: for each fractional phase, sum of coefficients = 1.0
        // This reduces gain variation across fractional delays
        for (int ph = 0; ph < kTableRes; ++ph)
        {
            double f = (double) ph / (double) kTableRes;
            double sum = 0.0;
            double u = (double) (f + (double) (kHalf - 1));
            for (int j = 0; j < 2 * kHalf; ++j, u -= 1.0)
            {
                double xi = (u + (double) kHalf) * (double) kTableRes;
                if (xi <= 0.0 || xi >= (double)(size - 2))
                    continue;
                int ix = (int) xi;
                double fr = xi - (double) ix;
                if (ix >= 0 && ix < size - 1)
                {
                    float v0 = table[(size_t) ix];
                    float v1 = table[(size_t) (ix + 1)];
                    sum += (double) v0 + (double)(v1 - v0) * fr;
                }
            }
            if (sum > 1e-12)
                phaseNorm[(size_t) ph] = (float) (1.0 / sum);
            else
                phaseNorm[(size_t) ph] = 1.0f;
        }
    }

    std::vector<float> table, buf;
    std::vector<float> phaseNorm = std::vector<float>(kTableRes, 1.0f);
    double step = 1.0, scale = 1.0, pos = 0.0;
    int hs = 1, len = 0;
};

/**
    host rate -> 48 kHz -> (model step per sample) -> host rate, mono.
    Same interface as before: prepare / release / reset / getLatencySamples / process.
*/
class FixedRateResampler
{
public:
    static constexpr double kModelRate = 48000.0;
    static constexpr int kChunk = 128;
    static constexpr int kRingSize = 1 << 15;
    static constexpr int kRingMask = kRingSize - 1;

    /** Fraction of the lower Nyquist where the filter's -6 dB point sits. */
    static constexpr double kCutoff = 0.97;

    void prepare (double hostRate)
    {
        hostRateStored = hostRate;
        maxModelBlock = (int) std::ceil ((double) kChunk * kModelRate / hostRate) + 4;
        maxHostBlock = (int) std::ceil ((double) maxModelBlock * hostRate / kModelRate) + 4;

        toModel.prepare (hostRate, kModelRate, kChunk, kCutoff);
        toHost.prepare (kModelRate, hostRate, maxModelBlock, kCutoff);

        modelBuf.assign ((size_t) maxModelBlock, 0.0f);
        hostOut.assign ((size_t) maxHostBlock, 0.0f);
        ring.assign ((size_t) kRingSize, 0.0f);

        primeZeros = measureWorstCaseDeficit() + 2;
        active = true;
        reset();
    }

    void release() { active = false; }

    void reset()
    {
        toModel.reset();
        toHost.reset();
        std::fill (ring.begin(), ring.end(), 0.0f);
        rd = 0;
        wr = primeZeros;
        count = primeZeros;
        underruns = 0;
    }

    int getLatencySamples() const noexcept { return primeZeros; }
    int getUnderrunCount() const noexcept { return underruns; }

    /** `in` and `out` may be the same pointer. `step` runs the model at 48 kHz. */
    template <typename Step>
    void process (const float* in, float* out, int numSamples, Step&& step)
    {
        int done = 0;

        while (done < numSamples)
        {
            const int n = std::min (kChunk, numSamples - done);

            const int m = toModel.process (in != nullptr ? in + done : zeros(), n,
                                           modelBuf.data(), maxModelBlock);

            for (int i = 0; i < m; ++i)
                modelBuf[(size_t) i] = step (modelBuf[(size_t) i]);

            const int k = toHost.process (modelBuf.data(), m, hostOut.data(), maxHostBlock);
            pushHost (hostOut.data(), k);

            for (int i = 0; i < n; ++i)
                out[done + i] = popHost();

            done += n;
        }
    }

private:
    const float* zeros()
    {
        if (zeroBuf.size() < (size_t) kChunk)
            zeroBuf.assign ((size_t) kChunk, 0.0f);
        return zeroBuf.data();
    }

    void pushHost (const float* p, int k) noexcept
    {
        for (int i = 0; i < k; ++i)
        {
            if (count >= kRingSize) { rd = (rd + 1) & kRingMask; --count; }
            ring[(size_t) wr] = p[i];
            wr = (wr + 1) & kRingMask;
            ++count;
        }
    }

    float popHost() noexcept
    {
        if (count == 0) { ++underruns; return 0.0f; }
        const float v = ring[(size_t) rd];
        rd = (rd + 1) & kRingMask;
        --count;
        return v;
    }

    /** One-sample-at-a-time simulation: max over time of (host in - host out). */
    int measureWorstCaseDeficit()
    {
        toModel.reset();
        toHost.reset();

        float zero = 0.0f;
        long long consumed = 0, produced = 0, worst = 0;
        const long long total = (long long) (hostRateStored * 0.25);

        while (consumed < total)
        {
            ++consumed;
            const int m = toModel.process (&zero, 1, modelBuf.data(), maxModelBlock);
            if (m > 0)
                produced += toHost.process (modelBuf.data(), m, hostOut.data(), maxHostBlock);
            worst = std::max (worst, consumed - produced);
        }

        return (int) worst;
    }

    KaiserSincResampler toModel, toHost;
    std::vector<float> modelBuf, hostOut, ring, zeroBuf;

    double hostRateStored = 48000.0;
    int maxModelBlock = 0, maxHostBlock = 0;
    int primeZeros = 0;
    int rd = 0, wr = 0, count = 0, underruns = 0;
    bool active = false;
};

} // namespace Crunchypy
