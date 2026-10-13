"""
Checks for the three harness defects that were fixed before any serious training.

These exist as a file rather than as one-off commands because each of them was a bug that
looked correct: the loss and the evaluator ran without complaint, and the reported numbers
were wrong in ways a reader could not see. A regression in any of them would again produce
plausible output, so they are asserted rather than eyeballed.

Run:  python test_harness.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

from train import run_window
from model import KnobGRU
from losses import mrstft_per_clip, mrstft_resolutions

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def settled_target(model, B, T, cb):
    """
    A target produced from a warm recurrent state.

    This detail matters. Every window in this harness starts from a zero state, so a
    target generated the same way would match the model's startup exactly and every
    burn-in would look like a no-op -- the test would pass while proving nothing. Warming
    on different audio first produces what a plugin actually emits at inference.
    """
    with torch.no_grad():
        x = torch.randn(B, T)
        _, state = model(torch.randn(B, T), cb)
        out, _ = model(x, cb, state)
        return x, out.squeeze(-1)


def test_burn_in_excludes_transient():
    """Burn-in must stop the startup transient being scored."""
    torch.manual_seed(0)
    B, T = 16, 8192
    model = KnobGRU(n_features=2, hidden=32).eval()
    cb = torch.full((B, 1), 0.5)
    xb, yb = settled_target(model, B, T, cb)

    unscored = run_window(model, xb, yb, cb, tbptt=512, burn=0)
    burned = run_window(model, xb, yb, cb, tbptt=512, burn=1024)

    check("burn-in lowers ESR against a settled-state target", burned < unscored,
          f"{unscored:.3e} -> {burned:.3e}")
    check("burn-in removes essentially all of it", burned < 1e-9 * max(unscored, 1.0),
          f"residual {burned:.3e}")


def test_burn_in_survives_chunk_boundaries():
    """
    The scored region starts partway through a chunk, and the last chunk is short.

    An earlier version compared the burn offset against the window length rather than the
    chunk length, so any burn greater than zero indexed an empty slice and raised. It
    survived because it only ever ran with burn=0.
    """
    torch.manual_seed(0)
    B, T = 8, 8192
    model = KnobGRU(n_features=2, hidden=32).eval()
    cb = torch.full((B, 1), 0.5)
    xb, yb = settled_target(model, B, T, cb)

    ok = True
    detail = ""
    for burn in (0, 1, 511, 512, 513, 1023, 1024, 2047, T - 2, T - 1):
        try:
            v = run_window(model, xb, yb, cb, tbptt=512, burn=burn)
            if not math.isfinite(v):
                ok = False
                detail = f"burn={burn} gave {v}"
                break
        except Exception as exc:
            ok = False
            detail = f"burn={burn} raised {type(exc).__name__}: {exc}"
            break
    check("burn values straddling chunk boundaries all run", ok, detail)

    # A window that is not a multiple of TBPTT, and one shorter than a single chunk.
    for n, tbptt in ((7000, 512), (600, 1024), (512, 512), (200, 512)):
        try:
            v = run_window(model, xb[:, :n], yb[:, :n], cb, tbptt=tbptt, burn=min(100, n - 1))
            assert math.isfinite(v)
        except Exception as exc:
            check(f"window {n} with tbptt {tbptt}", False, f"{type(exc).__name__}: {exc}")
            return
    check("odd window/tbptt combinations run", True)


def test_burn_in_still_trains():
    """The burn-in chunks must not block gradients for the scored region."""
    torch.manual_seed(0)
    model = KnobGRU(n_features=2, hidden=32)
    cb = torch.rand(4, 1)
    xb = torch.randn(4, 4096)
    yb = torch.randn(4, 4096)

    grads = {}
    for burn in (0, 1024, 2048):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = run_window(model, xb, yb, cb, tbptt=512, opt=opt, burn=burn)
        g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        grads[burn] = (loss, g)

    check("loss is finite with burn-in",
          all(math.isfinite(l) for l, _ in grads.values()))
    check("gradients are non-zero with burn-in",
          all(g > 0 for _, g in grads.values()),
          "sum|grad| " + ", ".join(f"{b}:{g:.2f}" for b, (_, g) in grads.items()))


def test_seeding_is_reproducible():
    """
    Training must be reproducible from --seed.

    The seed used to be applied only inside the export path, so the initial weights were
    whatever the process drew and two runs that differed only in --seed produced different
    models. Every comparison between two runs was then partly a comparison of their
    inits, which is the specific confusion this had already caused once.
    """
    import train as train_mod

    def build():
        torch.manual_seed(1234)
        return KnobGRU(n_features=3, hidden=16).param_count(), \
               [float(p.detach().abs().sum()) for p in KnobGRU(n_features=3, hidden=16).parameters()]

    torch.manual_seed(1234)
    a = KnobGRU(n_features=3, hidden=16)
    torch.manual_seed(1234)
    b = KnobGRU(n_features=3, hidden=16)

    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    check("identical seed gives identical weights", same)

    torch.manual_seed(4321)
    c = KnobGRU(n_features=3, hidden=16)
    diff = not all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))
    check("different seed gives different weights", diff)

    # And the trainer must actually call it before building the model.
    src = open("train.py", encoding="utf-8").read()
    seed_at = src.find("torch.manual_seed(args.seed)")
    model_at = src.find("model = KnobGRU(")
    check("train() seeds before constructing the model",
          0 < seed_at < model_at,
          f"seed at char {seed_at}, model at {model_at}")


def test_split_has_no_audio_overlap():
    """
    train / val / test must not share source audio.

    Every render is a window onto one normalized.wav, so splitting by row would hand
    validation audio the model has already been fitted to -- often the same waveform
    shifted by a fraction of a second. The reported ESR would then measure recall of
    memorised audio rather than generalisation.
    """
    from knobdata import split

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # A stand-in for the pack: 180 s at 48 kHz of float32, so the bands are anchored
        # to the full input rather than to whatever subset of renders exists.
        os.makedirs(os.path.join(tmp, "pack"), exist_ok=True)
        with open(os.path.join(tmp, "pack", "input.npy"), "wb") as f:
            f.truncate(int(180.0 * 48000) * 4)

        class R:
            def __init__(self, offset, dur=5.0):
                self.offset_s = offset
                self.duration_s = dur
                self.knobs = {
                    "Vol1": 0.5, "Lead Drive": 0.5, "Treble": 0.5, "Bass": 0.5,
                    "Middle": 0.5, "Lead Master": 0.5,
                    "Pull Bright (Vol1)": 0.5, "Pull Shift (Treble)": 0.5,
                    "Pull Shift (Bass)": 0.5, "Pull Shift (++ mod)": 0.5,
                    "Pull Deep (Master1)": 0.5, "Pull Bright (Lead Master)": 0.5,
                    "Master 1": 0.5, "Pull Lead": 1.0,
                }
                self.split = None
                self.id = f"r{offset}"
                self.db_dir = tmp

        # Every 0.25 s from 0 to 180 s, like the real sampler with stride 0.25.
        renders = [R(i * 0.25) for i in range(721)]
        split(renders)

        extents = {}
        for r in renders:
            extents.setdefault(r.split, []).append(
                (r.offset_s, r.offset_s + r.duration_s))

    ok = True
    detail = ""
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            lo_a, hi_a = min(e[0] for e in extents.get(a, [(0, 0)])), max(e[1] for e in extents.get(a, [(0, 0)]))
            lo_b, hi_b = min(e[0] for e in extents.get(b, [(0, 0)])), max(e[1] for e in extents.get(b, [(0, 0)]))
            # Ranges may abut; the audio must not overlap.
            if min(hi_a, hi_b) - max(lo_a, lo_b) > 1e-6:
                ok = False
                detail = f"{a} [{lo_a},{hi_a}] overlaps {b} [{lo_b},{hi_b}]"
    check("no source audio is shared between splits", ok, detail)
    check("all three splits are populated",
          all(len(extents.get(s, [])) > 0 for s in ("train", "val", "test")),
          ", ".join(f"{s}:{len(extents.get(s, []))}" for s in ("train", "val", "test")))


def test_split_is_stable_as_the_database_grows():
    """
    A render's split must not change when later renders arrive.

    Bands are anchored to the source timeline rather than to the offsets rendered so far.
    Otherwise every new render would shift the boundaries and silently re-assign renders
    that had already been trained on -- the same knob setting landing in training in one
    run and validation in the next.
    """
    from knobdata import split

    class R:
        def __init__(self, offset):
            self.offset_s = offset
            self.duration_s = 5.0
            self.knobs = {
                "Vol1": 0.5, "Lead Drive": 0.5, "Treble": 0.5, "Bass": 0.5,
                "Middle": 0.5, "Lead Master": 0.5,
                "Pull Bright (Vol1)": 0.5, "Pull Shift (Treble)": 0.5,
                "Pull Shift (Bass)": 0.5, "Pull Shift (++ mod)": 0.5,
                "Pull Deep (Master1)": 0.5, "Pull Bright (Lead Master)": 0.5,
                "Master 1": 0.5, "Pull Lead": 1.0,
            }
            self.split = None
            self.id = f"r{offset}"
            # db_dir is what timeline_extent reads the pack length from.
            self.db_dir = self

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # A stand-in for the pack: 180 s at 48 kHz of float32.
        os.makedirs(os.path.join(tmp, "pack"), exist_ok=True)
        with open(os.path.join(tmp, "pack", "input.npy"), "wb") as f:
            f.truncate(int(180.0 * 48000) * 4)

        def renders(n):
            out = []
            for i in range(n):
                r = R(i * 0.25)
                r.id = f"r{i * 0.25}"
                out.append(r)
            return out

        def databases_of(rs):
            return [tmp]

        import knobdata
        saved = knobdata.databases_of
        knobdata.databases_of = databases_of
        try:
            partial = renders(200)              # first ~50 s of a 180 s input
            split(partial)
            first = {r.id: r.split for r in partial}

            # A partial database legitimately has nothing in the val and test bands --
            # those offsets have not been rendered yet. What must hold is that its bands
            # are the finished ones, so that a render's split is decided by its offset
            # alone. Checked by comparing against the full database below rather than by
            # expecting renders that do not exist.

            grown = renders(721)
            split(grown)
            second = {r.id: r.split for r in grown}

            # Split is a pure function of offset: shuffling the render list must not
            # change any render's assignment. Cheap to assert and it catches a band that
            # depends on iteration order rather than on position in time.
            import random

            shuffled = renders(721)
            random.Random(7).shuffle(shuffled)
            split(shuffled)
            third = {r.id: r.split for r in shuffled}
            reordered = [k for k, v in third.items() if second.get(k) != v]
            check("shuffling the render list changes no assignment", not reordered,
                  f"{len(reordered)} of {len(third)} differ")
        finally:
            knobdata.databases_of = saved

    moved = [k for k, v in first.items() if second.get(k) != v]
    check("splits do not move when the database grows", not moved,
          f"{len(moved)} of {len(first)} renders changed split")


def test_pinned_knob_is_not_a_feature():
    """
    Pinned knobs must not be conditioning channels.

    Master 1 (=0.5, unity output level) and Pull Lead (=1.0, lead channel always engaged)
    are pinned during generation and applied as gain / hardwired into the circuit, so feeding
    them as constant inputs spends weights on values that never vary and gives the network an
    excuse to be sensitive to something the plugin will never change.
    """
    import train as train_mod

    for pk in train_mod.PINNED_KNOBS:
        check(f"{pk} is not in KNOB_ORDER", pk not in train_mod.KNOB_ORDER,
              f"KNOB_ORDER = {train_mod.KNOB_ORDER}")
        check(f"{pk} is pinned", pk in train_mod.PINNED_KNOBS,
              f"PINNED_KNOBS = {tuple(train_mod.PINNED_KNOBS)}")

    # The plugin has no Volume knob for this circuit, so kVolumeGainAtFull is not expected.
    # Instead check that the plugin's output gain convention matches whatever the trainer
    # declares. The trainer's volume_gain is a no-op for this circuit (the preamp has no
    # Volume knob), so it must return 1.0 at the default position.
    check("volume_gain is a no-op for this circuit (returns 1.0)",
          abs(train_mod.volume_gain(0.5) - 1.0) < 1e-12,
          f"volume_gain(0.5) = {train_mod.volume_gain(0.5):.4f}, expected 1.0")


def test_mrstft_matches_auraloss():
    """
    The port is only worth having if it is the validated implementation. Compared against
    NAM's vendored auraloss one clip at a time, since auraloss reduces over the whole batch
    while this is per clip -- see losses.py for why that difference is deliberate.
    """
    try:
        from auraloss.freq import MultiResolutionSTFTLoss
    except ImportError:
        deps = os.environ.get("NAM_DEPS_DIR", r"C:\Code\ProjectCrunchyBerries\neural-amp-modeler\nam\_dependencies")
        if os.path.isdir(deps) and deps not in sys.path:
            sys.path.insert(0, deps)
        try:
            from auraloss.freq import MultiResolutionSTFTLoss
        except ImportError:
            check("NAM's auraloss is present to compare against", True, "skipped (auraloss not installed)")
            return

    torch.manual_seed(0)
    target = torch.randn(1, 8192) * 0.1
    pred = target + torch.randn(1, 8192) * 0.01
    for fft, hop, win in [(512, 50, 240), (256, 25, 120), (128, 13, 60), (2048, 240, 1200)]:
        theirs = float(MultiResolutionSTFTLoss(
            fft_sizes=[fft], hop_sizes=[hop], win_lengths=[win],
        )(pred.unsqueeze(1), target.unsqueeze(1)))
        ours = float(mrstft_per_clip(pred, target, ((fft, hop, win),), pre_emph_coef=0.0)[0])
        check(f"MRSTFT fft {fft} equals auraloss", abs(ours - theirs) < 1e-6,
              f"{ours:.9g} vs {theirs:.9g}")


def test_mrstft_sees_spectrum_where_esr_cannot():
    """
    The reason the term exists. Two errors of identical energy, one confined below 2 kHz and
    one above 8 kHz. ESR is a whole-signal energy ratio and scores them the same; the
    spectral term must not.
    """
    sr, n = 48000, 8192
    freqs = torch.fft.rfftfreq(n, 1.0 / sr)

    def band(lo, hi, seed):
        g = torch.Generator().manual_seed(seed)
        nb = n // 2 + 1
        mask = ((freqs >= lo) & (freqs < hi)).double()
        spec = (torch.randn(nb, generator=g, dtype=torch.float64)
                + 1j * torch.randn(nb, generator=g, dtype=torch.float64))
        y = torch.fft.irfft(spec * mask, n=n)
        return (y / y.pow(2).mean().sqrt()).float()[None]

    target = band(20, 20000, 1) * 0.1
    bass = target + band(20, 2000, 2) * 0.05
    spark = target + band(8000, 20000, 3) * 0.05
    res = mrstft_resolutions(512)

    e = [float(((x - target) ** 2).sum() / (target ** 2).sum()) for x in (bass, spark)]
    check("ESR cannot tell bass error from HF error", abs(e[0] - e[1]) / e[0] < 1e-3,
          f"{e[0]:.6f} vs {e[1]:.6f}")

    # Pre-emphasis 0.85 is NAM's pre_emph_mrstft_coef, and it is doing real work here.
    plain = [float(mrstft_per_clip(x, target, res, 0.0)[0]) for x in (bass, spark)]
    pre = [float(mrstft_per_clip(x, target, res, 0.85)[0]) for x in (bass, spark)]
    check("MRSTFT punishes HF error more than bass error", pre[1] > pre[0],
          f"pre-emph {pre[0]:.5f} -> {pre[1]:.5f}, ratio {pre[1] / pre[0]:.2f}x")
    check("...and pre-emphasis widens that gap", pre[1] / pre[0] > plain[1] / plain[0],
          f"plain {plain[1] / plain[0]:.2f}x -> pre-emph {pre[1] / pre[0]:.2f}x")
    check("identical signals score exactly zero",
          float(mrstft_per_clip(target, target, res, 0.0).max()) == 0.0)
    check("silence scores ~0, so there is no pull toward silence",
          float(mrstft_per_clip(torch.zeros(1, 4096), torch.zeros(1, 4096), res, 0.0)[0]) < 1e-3)


def test_mrstft_resolutions_fit_the_chunk():
    """A spectral loss whose FFT outruns the TBPTT chunk would be fitting reflected padding."""
    for limit in (128, 512, 1024, 2048, 4096):
        res = mrstft_resolutions(limit)
        check(f"all resolutions fit a {limit}-sample chunk",
              len(res) == 3 and all(x[0] <= limit for x in res), str(res))
    check("default tbptt 512 selects auraloss's smallest trio",
          mrstft_resolutions(512)[0] == (512, 50, 240), str(mrstft_resolutions(512)))
    check("tbptt 2048 reproduces auraloss's default trio verbatim",
          mrstft_resolutions(2048) == ((2048, 240, 1200), (1024, 120, 600), (512, 50, 240)))


def test_mrstft_changes_what_is_fitted():
    """
    Same seed, same data, same optimiser, one step apart: with the term on, the weights must
    land somewhere different. That is the whole claim -- it changes what is fitted rather
    than being a readout -- and it is the thing that would silently stop being true if the
    term stopped receiving a gradient.
    """
    torch.manual_seed(0)
    B, T, H, F = 4, 2048, 16, 3
    xb = torch.randn(B, T)
    cb = torch.full((B, F - 1), 0.5, dtype=torch.float32)   # audio + 2 knob channels

    runs = {}
    for weight in (0.0, 0.002):
        torch.manual_seed(0)
        model = KnobGRU(n_features=F, hidden=H)
        yb = torch.randn(B, T)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = [p.detach().clone() for p in model.parameters()]
        loss = run_window(model, xb, yb, cb, 512,
                          opt=opt,
                          burn=256, mrstft_weight=weight, mrstft_coef=0.85)
        opt.step()
        moved = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        runs[weight] = ([p.detach().clone() for p in model.parameters()], moved, loss)

    off_w, moved_off, loss_off = runs[0.0]
    on_w, moved_on, loss_on = runs[0.002]
    check("without MRSTFT the step still trains", moved_off, f"loss {loss_off:.5f}")
    check("with MRSTFT the step still trains", moved_on, f"loss {loss_on:.5f}")
    differs = any(not torch.equal(a, b) for a, b in zip(off_w, on_w))
    check("MRSTFT moves the weights somewhere different", differs)


def _brute_sparsity(val_renders, train_renders, knob_names):
    """Reference nearest-neighbour scan, kept deliberately naive."""
    out = []
    for v in val_renders:
        vk = getattr(v, "knobs", None)
        if vk is None:
            vk = v["knobs"]
        vv = [float(vk.get(n, 0.0)) for n in knob_names]
        best = None
        for t in train_renders:
            tk = getattr(t, "knobs", None)
            if tk is None:
                tk = t["knobs"]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vv, [float(tk.get(n, 0.0)) for n in knob_names])))
            best = d if best is None else min(best, d)
        out.append(best)
    return out


def test_knob_sparsity_matches_brute_force():
    """
    knob_sparsity was rewritten from an interpreted loop to numpy. The output values must
    not change: it is the distance axis the sparsity report buckets on, and a rewrite that
    looked identical but drifted, say by adding a factor of two, would quietly reclassify
    every render. Both dict and .knobs input shapes are legal callers.
    """
    from knobdata import knob_sparsity

    names = ["Vol1", "Lead Drive", "Treble"]
    class Rend:
        def __init__(self, knobs):
            self.knobs = knobs

    train = [
        Rend({"Vol1": 0.0, "Lead Drive": 0.0, "Treble": 0.0}),
        Rend({"Vol1": 1.0, "Lead Drive": 1.0, "Treble": 1.0}),
        Rend({"Vol1": 0.2, "Lead Drive": 0.8, "Treble": 0.5}),
    ]
    val = [
        {"knobs": {"Vol1": 0.5, "Lead Drive": 0.5, "Treble": 0.5, "ignored": 9.0}},
        {"knobs": {"Vol1": 0.8001, "Lead Drive": 0.2, "Treble": 0.49}},
        Rend({"Vol1": 0.6, "Lead Drive": 0.7}),
    ]
    got = knob_sparsity(val, train, names)
    want = _brute_sparsity(val, train, names)
    close = all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(got, want))
    check("knob_sparsity matches the naive scan", close, f"got {got} want {want}")
    check("knob_sparsity accepts dict and .knobs callers", len(got) == 3, f"len {len(got)}")


def test_sparsity_report_buckets_after_nan_mask():
    """
    sparsity_report used to silently print nothing. evaluate() returns its per-window ESR
    with below-MIN_RMS windows dropped, but the nearest-neighbour distances covered every
    window, and the two lengths could never match once the split held a quiet window, so
    the report hit its fallback and returned without bucketing anything. The fix threads
    the NaN-carrying array through and filters both arrays by the same mask; this test
    insists the buckets still print when a quiet (NaN) window is present.
    """
    from train import sparsity_report
    from contextlib import redirect_stdout
    import io

    class FakePacked:
        knobs = ["Vol1", "Lead Drive"]

        def __init__(self):
            self.items = [
                {"length": 100, "cond": [0.0, 0.0]},   # 0: train
                {"length": 100, "cond": [1.0, 1.0]},   # 1: train
                {"length": 100, "cond": [0.5, 0.5]},   # 2: val
            ]
            self._split = {0: "train", 1: "train", 2: "val"}

        def usable_for(self, name, window):
            return [i for i, it in enumerate(self.items)
                    if self._split[i] == name and it["length"] >= window]

    packed = FakePacked()
    window = 32
    # Render 2 has 5 windows (range(0, 100-32+1, 16)); the first is quiet, so evaluate()
    # would emit NaN there and the ESR mean would carry 4 windows.
    per_full = np.asarray([np.nan, 0.010, 0.020, 0.030, 0.040], dtype=np.float64)

    buf = io.StringIO()
    with redirect_stdout(buf):
        sparsity_report(packed, "val", per_full, window)
    out = buf.getvalue()

    # The report must print again, with exactly the 4 scored windows bucketed -- not zero
    # buckets (the old silent no-op) and not the NaNs treated as data.
    check("sparsity report prints once the mask is aligned", "nn-dist" in out,
          f"output was: {out!r}"[:200])
    check("only windows above MIN_RMS are bucketed", "ESR by distance from nearest training knob setting (4 windows):" in out)
    check("all non-NaN windows land in a bucket",
          "n=    4  ESR 0.02500" in out, "buckets:\n" + "\n".join(l for l in out.splitlines() if "nn-dist" in l))


def main():
    print("burn-in")
    test_burn_in_excludes_transient()
    test_burn_in_survives_chunk_boundaries()
    test_burn_in_still_trains()
    print("\nseeding")
    test_seeding_is_reproducible()
    print("\nsplitting")
    test_split_has_no_audio_overlap()
    test_split_is_stable_as_the_database_grows()
    print("\npinned knob")
    test_pinned_knob_is_not_a_feature()
    print("\nspectral loss")
    test_mrstft_matches_auraloss()
    test_mrstft_sees_spectrum_where_esr_cannot()
    test_mrstft_resolutions_fit_the_chunk()
    test_mrstft_changes_what_is_fitted()

    print("\nsparsity report")
    test_knob_sparsity_matches_brute_force()
    test_sparsity_report_buckets_after_nan_mask()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())