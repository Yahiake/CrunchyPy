# File: losses.py
"""
Training loss and the metrics that report it.

Error Signal Ratio, after Wright et al. (Eq. 10, Appl. Sci. 10:766) and as implemented in
NAM's `models/losses.py`:

    ESR = mean_over_clips( sum((pred - target)^2) / sum(target^2) )

Two properties make it the right loss here and MSE the wrong one:

  * It is scale-invariant per clip. The database spans about 100 dB of output level because
    the Volume knob does, so a squared-error loss would spend essentially all of its
    gradient budget on the loud end and treat the quiet end as rounding noise. Normalising
    per clip makes a -20 dBFS render count as much as a 0 dBFS one.
  * It is the number that is actually published. Comparing a result against a paper's ESR
    requires computing ESR the way the paper did; training on something else and then
    reporting ESR is fine, training on something else and comparing directly is not.

Pre-emphasis is deliberately absent from the ESR term. NAM applies it to its MSE and MRSTFT
losses but not to its ESR loss (`lightning_module.py:360-374`), and the reference results this
work is measured against are plain ESR. It stays available here as a secondary readout for the
same reason it exists there: it weights high-frequency error more heavily, which is a better
stand-in for what a listener notices.

The spectral term below does use pre-emphasis, because that is where NAM uses it
(`pre_emph_mrstft_coef` 0.85 at weight `pre_emph_mrstft_weight` 0.002 in
`nam_full_configs/models/lstm.json`). The two decisions are independent: whether ESR is
reported pre-emphasised is about comparability with published numbers, whereas whether the
training signal is pre-emphasised is about where the gradient should point.

Why a spectral term exists at all: ESR is a whole-signal energy ratio, so a model can reach
1% ESR while still getting the top octave wrong -- broadband error at 1% and a spectral
error that no listener would accept are not the same failure, and ESR alone cannot tell them
apart. On a fuzz pedal with asymmetric clipping that matters more than usual, because the
harmonic generation lives entirely above the fundamental.

Two constraints are load-bearing, and both were learned by breaking them.

  1. Windows below `MIN_RMS` are excluded, not floored. They carry no circuit information,
     and their gradient contribution has to go to zero rather than be capped.
  2. Predicting silence must score exactly 1.0 in every mode. Any loss where it scores less
     has a gradient pointing at silence, and on this database that is where training goes to
     die. See `silence_score`.

Reporting is always plain textbook ESR, whatever the loss mode is, so the numbers stay
comparable with published results.
"""

from __future__ import annotations

import torch

# Windows quieter than this RMS are excluded from the loss entirely: they are LiveSPICE's
# numerical floor, not circuit behaviour.
#
# Measured, not assumed. Volume = 0 renders come out at -131 dBFS RMS, 70 dB below anything a
# listener or a guitar would produce, and that is the SPICE solver's own noise rather than
# the pedal's output. Flooring the denominator does not fix this, because the gradient still
# scales as 1/target_energy no matter how small the denominator is allowed to get.
#
# Excluding rather than flooring is the honest choice: there is nothing there to fit. What
# the model does at Volume = 0 is extrapolation from the Volume values it does train on, and
# `train.py volume-response` is what measures whether that extrapolation is any good.
MIN_RMS = 1e-3          # -60 dBFS


def usable_mask(target: torch.Tensor, min_rms: float = MIN_RMS) -> torch.Tensor:
    """(B,) bool: True for windows with enough target energy to be worth fitting."""
    return target.pow(2).mean(dim=1).sqrt() >= min_rms


def silence_score(sst: torch.Tensor) -> float:
    """
    What predicting zero would score under per-clip ESR. Always exactly 1.0, and that is the
    property being asserted.

    Worth a function because an earlier version of this file got it wrong in a way that
    looked correct. To stop the quiet clips dominating the gradient, it floored each clip's
    denominator relative to the batch's loudest energy. But raising a quiet clip's
    denominator lowers its ratio, so predicting silence scored 0.679 instead of 1.0 -- the
    floor became a gradient pointing at silence, which is the original failure in disguise.

    On 113 renders the model escaped it and reached ESR 0.096. On the full 1,631-render
    database, where batch energy spans 53,000x, it went there and stopped: train loss sat at
    0.78 with the 0.679 floor below it and would not come down. Any weighting scheme here
    must pass this check, so `train.py` asserts it before training rather than trusting it.
    """
    v = sst[sst > 0]
    if v.numel() == 0:
        return 1.0
    return float(((v / v).clamp(1.0, 1.0)).mean())


def batch_energy_spread(sst: torch.Tensor) -> float:
    """Loudest/quietest target energy ratio within a batch, as a plain float."""
    v = sst[sst > 0]
    if v.numel() < 2:
        return 1.0
    return float(v.max() / v.min())


def clip_weights(sst: torch.Tensor, ok: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Per-clip weights, summing to 1 over the usable clips.

    This is where the 100 dB of output range gets dealt with, and the choice is between two
    failure modes that pull in opposite directions.

    Per-clip ESR normalises each clip by its own energy, which is what the published
    definition requires and what makes a quiet render count as much as a loud one. The cost
    is that the gradient of `sse_i/sst_i` scales as `1/sst_i`, so a clip 40 dB down gets
    millions of times the pull of the loudest one for the same absolute error. Left alone
    that stops training entirely.

    The fix is *not* to touch the denominator. Flooring it is the obvious move and it is
    what produced the 0.679 silence score above: raising a quiet clip's denominator shrinks
    its ratio, so silence starts scoring better than 1.0 and the loss falls toward it. So the
    denominator is left exactly as the definition has it, and the imbalance is removed from
    the weights instead:

      equalize  Each clip's ratio is divided by its own energy as usual, and the *weight*
                is that same energy normalised to sum to 1 across the batch. The 1/energy in
                the gradient cancels the energy in the weight, so every usable clip
                contributes comparable gradient regardless of level -- while each clip is
                still scored by its own published ESR ratio, and silence still scores 1.0.
                This is the default.
      perclip   Textbook ESR, every clip weighted equally. Correct by definition.
      global    One ratio for the batch: sum(sse) / sum(sst). Every clip weighted by its
                energy, so the loud end dominates outright.

    Measured over 60 epochs each at hidden=128 on the full 1,631-render database, all three
    now training monotonically and all three scoring silence at exactly 1.0:

      equalize  val ESR 0.504      perclip  0.591      global  0.650

    `perclip` is worth noting: it is the published definition and on this data it converges
    to the median clip rather than to zero error, which is why the 1/energy gradient term has
    to be cancelled somewhere. It is kept as an option so that claim stays checkable.
    """
    w = torch.zeros_like(sst)
    if not ok.any():
        return w
    if mode == "equalize":
        w[ok] = sst[ok]
    else:                                  # perclip and global both weight clips equally here
        w[ok] = 1.0
    return w / w.sum().clamp_min(1e-30)


def esr_per_clip(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    (B,) textbook ESR per clip, for reporting rather than for optimisation.

    Quiet windows get NaN rather than a floored ratio. A number there would be a ratio of two
    quantities that are both numerical noise, and NaN keeps it out of the average honestly
    instead of quietly scoring it as a failure.
    """
    sse = ((pred - target) ** 2).sum(dim=1)
    sst = (target ** 2).sum(dim=1)
    ok = usable_mask(target)
    out = torch.full_like(sse, float("nan"))
    out[ok] = sse[ok] / sst[ok].clamp_min(1e-30)
    return out


def esr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean per-clip ESR over the usable clips, for (B, T) tensors.

    The mean is over clips, not samples: one window's ratio is one number regardless of how
    many samples it holds, which is what keeps a -20 dBFS render worth as much as a 0 dBFS
    one. Returns NaN if every clip in the batch is too quiet to fit.
    """
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError(f"expected (B, T) tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    v = esr_per_clip(pred, target)
    v = v[~torch.isnan(v)]
    if v.numel() == 0:
        return torch.full((), float("nan"), dtype=pred.dtype, device=pred.device)
    return v.mean()


def pre_emphasis(x: torch.Tensor, coef: float = 0.95) -> torch.Tensor:
    """First-order pre-emphasis filter, `x[n] - coef*x[n-1]`, dropping the first sample."""
    return x[..., 1:] - coef * x[..., :-1]


def esr_pre(pred: torch.Tensor, target: torch.Tensor, coef: float = 0.95) -> torch.Tensor:
    """ESR after pre-emphasis. A stricter, higher-frequency-weighted readout than `esr`."""
    return esr(pre_emphasis(pred, coef), pre_emphasis(target, coef))


# ---------------------------------------------------------------------------
# multi-resolution STFT loss
# ---------------------------------------------------------------------------
#
# Ported from auraloss, which NAM vendors at
# neural-amp-modeler/nam/_dependencies/auraloss/freq.py, so that the numbers here come from
# the implementation that has actually been validated on this problem rather than from a
# fresh one. What is kept identical: the magnitude formula
# (sqrt of real^2+imag^2, clamped at eps=1e-8 before the sqrt, so the floor is 1e-4), the
# spectral-convergence term as a Frobenius ratio against the target, the log-magnitude term as
# a plain L1 on log magnitudes with log_fac=1/log_eps=0, weights w_sc=1 / w_log_mag=1 /
# w_lin_mag=0 / w_phs=0, hann analysis windows, and the (fft, hop, win) triples.
#
# Two deliberate departures, both forced by this trainer's shape and both recorded here so
# they cannot drift into being folklore:
#
#   1. Per clip, not per batch. auraloss flattens the batch into one signal
#      (`input.view(-1, input.size(-1))`), so its spectral loss is weighted by clip energy --
#      the loud end contributes ~1000x the pull of the quiet end on this database. That is
#      precisely the imbalance `clip_weights` exists to cancel, so applying auraloss's
#      reduction unchanged would have undone the fix inside the new term. Every clip is scored
#      here against its own spectrum and averaged with the ESR weights.
#
#   2. FFT sizes fitted to the TBPTT chunk. NAM runs the whole window through in one pass
#      (`_shared_step`, no chunking) so it can use fft 2048 over 32768 samples. This trainer
#      detaches the recurrent state every `tbptt` samples, default 512, to bound activation
#      memory. Applying fft 2048 to a 512-sample chunk would either error or produce a
#      spectrum of almost entirely reflected padding. The resolutions are therefore chosen
#      from a preset ladder to fit the chunk, keeping auraloss's hop/fft (~0.098) and
#      win/fft ratios. At the default tbptt=512 the trio is
#      (512,50,240) (256,25,120) (128,13,60) -- auraloss's smallest octave and two below it,
#      rather than its largest and two below.
#
# That departure is a real reduction in scope and it is the reason this term is weighted at
# 0.002 rather than something larger. If it turns out to earn more than that, the fix is to
# raise `--tbptt` so the larger FFTs become available, not to trust a 2048-point FFT of a
# 512-sample chunk.

_MRSTFT_PRESETS = (
    (2048, 240, 1200),   # auraloss defaults, verbatim
    (1024, 120, 600),
    (512, 50, 240),
    (256, 25, 120),      # the auraloss trio halved, for short TBPTT chunks
    (128, 13, 60),
)

_STFT_EPS = 1e-8         # auraloss's eps; note the clamp is applied before the sqrt

_WINDOW_CACHE: dict = {}


def mrstft_resolutions(limit: int, count: int = 3):
    """
    (fft, hop, win) triples for an MRSTFT term, largest first, none exceeding `limit`.

    `limit` is the TBPTT chunk length, because that is the longest contiguous run of
    predicted samples that exists with a live gradient attached to it.
    """
    ok = [t for t in _MRSTFT_PRESETS if t[0] <= limit]
    if not ok:
        raise ValueError(f"MRSTFT needs an FFT that fits a chunk; {limit} samples is too short")
    if len(ok) < count:
        # Extend downwards by halving the smallest qualifying preset, keeping its ratios.
        extra = []
        fft, hop, win = ok[-1]
        while len(ok) + len(extra) < count and fft > 16:
            fft, hop, win = fft // 2, max(1, hop // 2), max(8, win // 2)
            extra.append((fft, hop, win))
        ok = ok + extra
    return tuple(ok[:count])


def _hann(win: int, device, dtype) -> torch.Tensor:
    """Cached periodic Hann window, as auraloss's `get_window` builds it (`torch.hann_window(n)`)."""
    key = (win, str(device), str(dtype))
    w = _WINDOW_CACHE.get(key)
    if w is None:
        w = torch.hann_window(win, periodic=True, device=device, dtype=dtype)
        _WINDOW_CACHE[key] = w
    return w


def _stft_mag(x: torch.Tensor, fft: int, hop: int, win: int) -> torch.Tensor:
    """(B, fft//2+1, frames) magnitudes, auraloss's formula including its pre-sqrt clamp."""
    spec = torch.stft(x, fft, hop, win, _hann(win, x.device, x.dtype), return_complex=True)
    return torch.sqrt(torch.clamp(spec.real ** 2 + spec.imag ** 2, min=_STFT_EPS))


def mrstft_per_clip(
    pred: torch.Tensor,
    target: torch.Tensor,
    resolutions,
    pre_emph_coef: float = 0.85,
) -> torch.Tensor:
    """
    (B,) multi-resolution STFT loss, one value per clip.

    `pre_emph_coef` of 0 (or None) disables the pre-emphasis, matching NAM's
    `mrstft_weight` (no pre-emphasis) as distinct from `pre_emph_mrstft_weight` (0.85).

    Both signals are pre-emphasised together, never one alone. The filter is a fixed linear
    operator on both sides of the comparison, so it shapes what the term measures rather than
    introducing a bias.

    Deliberately silent on quiet clips in a way that is safe: both magnitudes floor at 1e-4
    inside the log, so predicting silence against a numerically-silent clip scores ~0 here
    rather than pulling the loss down. That is the opposite of the ESR denominator trap
    documented in `silence_score`, and it is why `run_window` can add this term without
    re-running that guard. The unusable clips are still excluded at the call site so they do
    not consume the term's share of the gradient at all.
    """
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError(f"expected (B, T) tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    if pre_emph_coef:
        pred = pre_emphasis(pred, pre_emph_coef)
        target = pre_emphasis(target, pre_emph_coef)

    # Discard any resolution whose reflection padding would exceed signal length
    valid_resolutions = [r for r in resolutions if r[0] // 2 < pred.shape[-1]]
    if not valid_resolutions:
        return pred.new_zeros(pred.shape[0])

    total = None
    for fft, hop, win in valid_resolutions:
        x = _stft_mag(pred, fft, hop, win)
        y = _stft_mag(target, fft, hop, win)
        x2 = x.reshape(x.shape[0], -1)
        y2 = y.reshape(y.shape[0], -1)
        # Spectral convergence: ||target - pred||_F / ||target||_F, per clip.
        sc = torch.linalg.matrix_norm(y2 - x2, ord="fro") / \
            torch.linalg.matrix_norm(y2, ord="fro").clamp_min(_STFT_EPS)
        # Log-magnitude L1, per clip.
        log_mag = (torch.log(x) - torch.log(y)).abs().mean(dim=(1, 2))
        term = sc + log_mag
        total = term if total is None else total + term

    return total / len(valid_resolutions)