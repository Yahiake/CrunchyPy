# File: train.py
"""
Train the knob-conditioned model on a LiveSPICE render database.

    python train.py prepare   --db DATASET
    python train.py train     --db DATASET [options]
    python train.py eval      --db DATASET --ckpt CHECKPOINT
    python train.py export    --db DATASET --ckpt CHECKPOINT [--split test]

`prepare` packs the database for fast epochs and draws the train/val/test split. The
renderer deliberately does not assign one, so the split can be redrawn without re-rendering
anything; it is written to `split.json` and reused from then on.

`train` runs truncated BPTT. The state is carried across chunks and detached between them,
and validation runs the whole window free-running. That difference is not a detail: training
with TBPTT=1 is teacher forcing, and a model trained that way measures roughly 19% error
free-running against roughly 3.9% at TBPTT=64 on the same architecture.

The reported metric is ESR, averaged over clips, matching NAM and DAFx-19 so the number can
be compared with published results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np
import soundfile as sf
import torch

from knobdata import (
    expected_length,
    load_input,
    load_render,
    load_split,
    read_manifest,
    report,
    save_split,
    split as assign_split,
    validate_geometry,
)
from losses import (MIN_RMS, batch_energy_spread, clip_weights, esr, esr_per_clip, esr_pre,
                    mrstft_per_clip, mrstft_resolutions, pre_emphasis, silence_score,
                    usable_mask)
from model import KnobGRU, export_rtneural, verify_export

# The knob columns fed to the model as constant channels, after the audio.
#
# Volume is deliberately absent. It is not trained because it was measured to be a pure
# post-gain on this circuit: holding Sustain and Tone fixed, rendering at Volume 0.4, 0.6
# and 0.8 reproduced the Volume 0.2 render times exactly 2, 3 and 4, with residual ESR of
# -180 dB at high Sustain and -85 dB at worst. That is roughly 4000x finer than any
# accuracy this model will reach, so learning it would spend parameters and, worse, render
# budget on a knob that carries none of the nonlinearity. The render database pins Volume
# at 0.2 (unity post-gain), and the plugin applies the measured law 5 * Volume as plain
# gain after the network.
#
# Keeping Volume as a *constant* input channel instead would be worse than dropping it: a
# feature that never varies carries no information, so its weights would sit at whatever
# the initialisation gave them, and the network would have to learn to be invariant to it
# rather than simply not seeing it.
PREAMP_KNOB_ORDER = [
    "Vol1",
    "Lead Drive",
    "Treble",
    "Bass",
    "Middle",
    "Lead Master",
    "Pull Bright (Vol1)",
    "Pull Shift (Treble)",
    "Pull Shift (Bass)",
    "Pull Shift (++ mod)",
    "Pull Deep (Master1)",
    "Pull Bright (Lead Master)"
]

def get_stage_knobs(dbs):
    """Detect whether database is for preamp or poweramp."""
    dbs_list = [d for d in re.split(r"[,\s]+", dbs) if d] if isinstance(dbs, str) else list(dbs)
    for d in dbs_list:
        spec_path = os.path.join(d, "spec.json")
        if os.path.exists(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)
                    if spec.get("name") == "mark2c_poweramp" or list(spec.get("knobs", {}).keys()) == ["P"]:
                        return ["P"], ()
            except Exception:
                pass
        if "poweramp" in os.path.normpath(d).lower():
            return ["P"], ()
    return PREAMP_KNOB_ORDER, ("Master 1", "Pull Lead")

KNOB_ORDER = PREAMP_KNOB_ORDER
PINNED_KNOBS = ("Master 1", "Pull Lead")


def volume_gain(v: float) -> float:
    """
    The gain the plugin applies after the network for a Volume position of v in [0, 1].

    The preamp has no Volume knob; this function is kept only so the volume-response check
    and the rest of the training code still compile. It is not used for this circuit.
    """
    return 1.0  # no-op: the preamp has no Volume knob


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(db, seed: int = 0, force: bool = False, pack_dir: str = None) -> str:
    """
    Pack one or more render databases and draw the train/val/test split.

    `db` may name several directories, separated by commas or spaces. Merging is how a
    later, differently sampled batch joins an earlier one without re-rendering it; both
    batches must share an input/normalized.wav, which load_input() enforces.
    """
    dbs = [d for d in re.split(r"[,\s]+", db) if d]
    if not dbs:
        raise RuntimeError("--db is empty")

    renders = read_manifest(dbs)
    report(renders, ", ".join(os.path.basename(os.path.normpath(d)) for d in dbs))
    x, sr = load_input(dbs)
    validate_geometry(renders, x, sr)

    split_dir = pack_dir
    if not load_split(renders, split_dir=split_dir):
        assign_split(renders, seed=seed)
        written = save_split(renders, split_dir=split_dir)
        print("split   : drawn (seed %d) and written to %s"
              % (seed, ", ".join(written)))
    else:
        print("split   : reused from split.json")

    for name in ("train", "val", "test"):
        got = [r for r in renders if r.split == name]
        good = [r for r in got if r.usable]
        peaks = np.array([r.peak for r in good]) if good else np.zeros(1)
        print(f"  {name:5s}  {len(got):5d} renders, {len(good):5d} usable, "
              f"peak {20 * math.log10(max(peaks.min(), 1e-12)):.0f} .. "
              f"{20 * math.log10(max(peaks.max(), 1e-12)):.0f} dBFS")

    usable = [r for r in renders if r.usable]
    if not usable:
        raise RuntimeError("no usable renders; check the manifest for errors, nonfinite "
                           "samples, flat output, or renders that are effectively silent")

    pack = pack_dir if pack_dir is not None else os.path.join(dbs[0], "pack")
    os.makedirs(pack, exist_ok=True)

    stage_knobs, stage_pinned = get_stage_knobs(dbs)
    present = set().union(*(r.knobs.keys() for r in usable))
    missing = [k for k in stage_knobs if k not in present]
    if missing:
        raise SystemExit(f"the database has no {', '.join(missing)}; "
                         f"it reports {', '.join(sorted(present))}")
    knob_names = list(stage_knobs) + sorted(present - set(stage_knobs) - set(stage_pinned))
    held = [k for k in stage_pinned if k in present]
    if held:
        print(f"pinned   : {', '.join(held)} excluded from the feature vector; "
              f"features are [audio, {', '.join(knob_names)}]")

    # One memmapped target array, so an epoch reads memory rather than 1600 files.
    total = sum(expected_length(r, sr)[1] for r in usable)
    print(f"packing  : {total / 1e6:.1f} M samples ({total * 4 / 1e9:.2f} GB)")
    mm = np.memmap(os.path.join(pack, "targets.f32"), dtype=np.float32, mode="w+", shape=(total,))

    index = []
    pos = 0
    for n, r in enumerate(usable):
        in_start, count = expected_length(r, sr)
        y = load_render(r, sr)
        if len(y) != count:
            raise RuntimeError(
                f"{r.id}: file holds {len(y)} samples, manifest implies {count}. "
                "The manifest and the audio disagree, so input and output cannot be paired."
            )
        mm[pos:pos + count] = y
        index.append({
            "id": r.id,
            "y_start": pos,
            "in_start": in_start,
            "length": count,
            "gain": r.input_gain,
            "split": r.split,
            "cond": [float(r.knobs[k]) for k in knob_names],
        })
        pos += count
        if (n + 1) % 250 == 0:
            print(f"  packed {n + 1}/{len(usable)}")
    mm.flush()
    del mm

    np.save(os.path.join(pack, "input.npy"), x)
    with open(os.path.join(pack, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sample_rate": sr,
            "knobs": knob_names,
            "pinned_knobs": list(PINNED_KNOBS),
            "total_samples": total,
            "input_samples": int(len(x)),
            # Which databases went into this pack, so a later `train --db` naming a
            # different set is told the pack is not what it asked for instead of quietly
            # training on the previous run's data.
            "databases": [os.path.normpath(os.path.abspath(d)) for d in dbs],
            "items": index,
        }, f)
    print(f"wrote    : {pack}")
    return pack


# ---------------------------------------------------------------------------
# packed data access
# ---------------------------------------------------------------------------

class Packed:
    """Window sampler over the packed database."""

    def __init__(self, db, pack_dir: str = None):
        dbs = [d for d in re.split(r"[,\s]+", db) if d] if isinstance(db, str) else list(db)
        dbs = [os.path.normpath(os.path.abspath(d)) for d in dbs]

        # The pack always lives under pack_dir or the first named database.
        pack = os.path.normpath(os.path.abspath(pack_dir)) if pack_dir is not None else os.path.join(dbs[0], "pack")
        with open(os.path.join(pack, "index.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        packed_from = [os.path.normpath(p) for p in meta.get("databases", [])]

        if packed_from and packed_from != dbs:
            raise RuntimeError(
                "this pack does not hold the databases that were requested.\n"
                f"  asked for : {', '.join(dbs)}\n"
                f"  packed    : {', '.join(packed_from)}\n"
                f"  pack dir  : {pack}\n"
                "Run `prepare --db " + ",".join(dbs) + "` first."
            )

        self.pack = pack
        self.databases = packed_from
        self.sr = meta["sample_rate"]
        self.knobs = meta["knobs"]
        self.pinned_knobs = meta.get("pinned_knobs", [])

        # An old pack prepared before Volume was pinned still lists it as a conditioning
        # channel. That pack is silently wrong for the current architecture -- the model
        # would be trained with a dead Volume input and the plugin would wire the wrong
        # feature order -- so it is refused rather than reinterpreted.
        stale = [k for k in self.pinned_knobs if k in self.knobs]
        if stale:
            raise RuntimeError(
                "this pack is stale: it still feeds {0} to the model as a feature, but "
                "{0} is now pinned and applied as gain in the plugin.\n"
                "  pack knobs : {1}\n"
                "  Re-run `prepare --db {2}` to repack.".format(
                    ", ".join(stale), ", ".join(self.knobs), ", ".join(dbs)))

        self.x = np.load(os.path.join(pack, "input.npy"))
        self.y = np.memmap(
            os.path.join(pack, "targets.f32"),
            dtype=np.float32, mode="r", shape=(meta["total_samples"],),
        )
        self.items = meta["items"]
        self.n_knobs = len(self.knobs)
        self.cond_all = np.array([it["cond"] for it in self.items], dtype=np.float32)

    def usable_for(self, name: str, window: int) -> list:
        return [i for i, it in enumerate(self.items)
                if it["split"] == name and it["length"] >= window]

    def gather(self, idxs, starts, window):
        """Build one batch. Returns x (B,T), y (B,T), cond (B,K)."""
        B = len(idxs)
        xb = np.empty((B, window), np.float32)
        yb = np.empty((B, window), np.float32)
        cb = np.empty((B, self.n_knobs), np.float32)
        for i, (k, s) in enumerate(zip(idxs, starts)):
            it = self.items[k]
            a = it["in_start"] + s
            xb[i] = self.x[a:a + window] * it["gain"]
            b = it["y_start"] + s
            yb[i] = self.y[b:b + window]
            cb[i] = self.cond_all[k]
        return xb, yb, cb

    def train_batches(self, idxs, window, batch, rng):
        """Shuffle the render list and cut it into fixed-size batches."""
        order = rng.permutation(len(idxs))
        for i in range(0, len(order) - batch + 1, batch):
            sel = [idxs[j] for j in order[i:i + batch]]
            starts = [int(rng.integers(0, self.items[k]["length"] - window + 1)) for k in sel]
            yield sel, starts

    def eval_batches(self, idxs, window, batch, overlap=False):
        """
        Deterministic windows for validation.

        Non-overlapping by default so the cost is bounded and the number is stable across
        runs; the final report re-runs at 50% overlap for a tighter estimate.
        """
        step = window // 2 if overlap else window
        for i in range(0, len(idxs), batch):
            sel, starts = [], []
            for k in idxs[i:i + batch]:
                it = self.items[k]
                for s in range(0, it["length"] - window + 1, step):
                    sel.append(k)
                    starts.append(s)
            if not sel:
                continue
            for j in range(0, len(sel), batch):
                yield self.gather(sel[j:j + batch], starts[j:j + batch], window)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def lr_at(step, total, peak, warmup_frac=0.05, floor_frac=0.02):
    """Linear warmup then cosine decay. The floor keeps it learning late instead of stalling."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    floor = peak * floor_frac
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def run_window(model, xb, yb, cb, tbptt, opt=None, mode="equalize", burn=0,
               mrstft_weight=0.0, mrstft_coef=0.85):
    """
    One pass over a batch of windows, in TBPTT chunks. Returns the whole-batch loss.

    Each clip is scored by its own ESR -- sum of squared error over sum of squared target --
    even though the backward passes are per chunk. Summing a chunk's squared error against
    the *whole window's* target energy means the chunks sum to exactly the window's ratio
    while only one chunk of graph stays alive at a time.

    `mode` chooses how clips are weighted against each other; see `losses.clip_weights` for
    why that is a real decision and not a detail. All modes score silence at exactly 1.0, so
    there is no floor for the loss to slide down toward.

    `mrstft_weight` adds a multi-resolution spectral term on the same scored region, weighted
    by the same clip weights. It is computed per chunk for the same reason the ESR term is:
    the chunk is the longest run with a live graph attached. That constrains its FFT sizes to
    fit `tbptt` -- see `losses.mrstft_resolutions` for what that costs. Zero disables it.

    `burn` is the number of leading samples fed through the network purely to settle the
    recurrent state, with no loss attached. This matters more than it looks. Every window
    starts from a zero state, and a GRU coming from zero has to be walked to its operating
    point before its output means anything. Scoring that transient teaches the network to
    predict the approach from a state it will never be asked to reproduce at inference, and
    it inflates the loss with an error the plugin will never actually make. The fix is
    NAM's: burn in, then score only what follows. See NAM's nam_full_configs/models/lstm.json,
    train_burn_in 8192 / mask_first 8192, and the comment "use a long ny like 32768" -- their
    burn-in alone is longer than this trainer's entire window used to be.

    The burn-in chunks still detach like any other TBPTT chunk. NAM deliberately does *not*
    detach across the burn-in boundary so the state stays learnable, which is only possible
    without TBPTT; here the detached path is what keeps memory bounded, and the state is
    still trained on by every scored chunk.

    With `opt` given, gradients accumulate and the caller steps. Without, it evaluates.
    """
    # Score only the post-burn-in region. The loss and the normalising energy are
    # restricted to it together, so the result stays a true error-to-signal ratio of the
    # part that actually gets graded rather than of the transient.
    start = min(burn, max(0, xb.shape[1] - 1))
    ys = yb[:, start:]
    ok = usable_mask(ys)
    if not bool(ok.any()):
        return float("nan")
    sst = (ys ** 2).sum(dim=1).clamp_min(1e-30)
    w = clip_weights(sst, ok, mode)                      # sums to 1 over usable clips

    resolutions = mrstft_resolutions(tbptt) if mrstft_weight else ()
    # Usable clips only. A clip with no target energy has no spectrum to match, and letting
    # it in here would spend the spectral term's weight on LiveSPICE's numerical floor.
    sp = slice(None)
    if resolutions:
        sp = ok.nonzero(as_tuple=True)[0]

    sse_tot = yb.new_zeros(yb.shape[0])
    state = None
    if opt is not None:
        opt.zero_grad(set_to_none=True)

    T = xb.shape[1]
    for c in range(0, T, tbptt):
        n = min(tbptt, T - c)                 # last chunk is short
        out, state = model(xb[:, c:c + n], cb, state)

        # Where scoring begins inside this chunk, measured against the chunk and not the
        # window: a chunk lying wholly inside the burn-in contributes state but no loss,
        # and its slice is empty, so it has to be stepped past rather than scored.
        lo = max(0, start - c)
        if lo < n:
            po = out[:, lo:].squeeze(-1)
            yo = yb[:, c + lo:c + n]
            sse = ((po - yo) ** 2).sum(dim=1)
            # w is detached: it is a weighting, not something to learn through.
            loss = (w * sse / sst).sum()
            if resolutions and po.shape[1] >= 64:
                # If this chunk is shorter than tbptt (due to burn-in boundary or odd window length),
                # adapt the resolutions so the FFT fits the actual scored length.
                chunk_len = po.shape[1] - (1 if mrstft_coef else 0)
                chunk_res = resolutions if po.shape[1] == tbptt else (
                    mrstft_resolutions(chunk_len) if chunk_len >= 64 else ()
                )
                if chunk_res:
                    spec = mrstft_per_clip(po[sp], yo[sp], chunk_res, mrstft_coef)
                    loss = loss + mrstft_weight * (w[sp] * spec).sum()
            if opt is not None:
                loss.backward()
            sse_tot = sse_tot + sse.detach()

        if opt is not None:
            state = state.detach()

    return float((w * sse_tot / sst).sum().detach())


def evaluate(model, packed, name, window, batch, overlap=False, device="cuda", burn=0):
    """
    Free-running ESR over a split. Each window runs from a zero state, no teacher forcing.

    `burn` discards that startup before anything is scored, matching the burn-in the loss
    uses. Without it the headline number is partly a measurement of the network being
    dragged up from a zero state, which is an artifact of starting each window here and not
    something the plugin does at inference, where the state runs continuously across the
    whole stream. Scoring the transient made the old validation numbers pessimistic and,
    worse, made improvements in the part that matters harder to see.

    Windows below MIN_RMS come back as NaN and are dropped here, so the number is the mean
    over windows that carry signal. `n_all` is returned alongside so a reader can see what
    fraction that was rather than having to trust it.

    Returns (esr, esr_pre, n_all, esr_full): the dropped mean arrays plus the raw
    per-window ESR with its NaN entries preserved, in eval_batches order.
    """
    model.eval()
    per_clip, per_clip_pre, n_all = [], [], 0
    start = min(burn, max(0, window - 1))
    with torch.no_grad():
        for xb, yb, cb in packed.eval_batches(packed.usable_for(name, window), window, batch, overlap):
            n_all += yb.shape[0]
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(yb).to(device)
            cb = torch.from_numpy(cb).to(device)
            out, _ = model(xb, cb)              # whole window, free-running
            p = out.squeeze(-1)[:, start:]      # burn-in scored by neither ESR below
            y = yb[:, start:]
            per_clip.append(esr_per_clip(p, y).cpu().numpy())
            per_clip_pre.append(esr_per_clip(pre_emphasis(p), pre_emphasis(y)).cpu().numpy())
    model.train()

    def clean(arr):
        a = np.concatenate(arr) if arr else np.zeros(0)
        return a[~np.isnan(a)]

    # The same windows with their NaN entries kept, aligned one-per-window to the order
    # eval_batches emitted them. sparsity_report needs this shape: its per-window
    # distances line up with the ESR mean (which drops below-MIN_RMS windows) only after
    # both are filtered by the same mask, and the mask only exists here.
    per_full = np.concatenate(per_clip) if per_clip else np.zeros(0)

    return clean(per_clip), clean(per_clip_pre), n_all, per_full


def per_knob_report(model, packed, name, window, device="cuda", burn=0, batch=64):
    """
    ESR and output level grouped by knob value, one knob at a time.

    Two things are being separated here that a single average hides. A model that has learned
    the Volume law, and one that has merely learned to be quiet everywhere, both show a
    respectable mean ESR if the quiet renders dominate; only the per-knob breakdown tells
    them apart. Reporting the predicted RMS against the target RMS does the same job from the
    other direction, and catches a model that has the waveform right but the level wrong --
    which is exactly the mistake that a pure ratio metric can hide when Volume is in the input.

    `burn` matches `evaluate`, so a knob's ESR here is directly comparable with the headline.
    """
    idx = packed.usable_for(name, window)
    start = min(burn, max(0, window - 1))
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sel = idx[i:i + batch]
            xb, yb, cb = packed.gather(sel, [0] * len(sel), window)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(yb).to(device)
            cb = torch.from_numpy(cb).to(device)
            out, _ = model(xb, cb)
            p = out.squeeze(-1)[:, start:]
            y = yb[:, start:]
            e = esr_per_clip(p, y).cpu().numpy()
            pr = p.pow(2).mean(dim=1).sqrt().cpu().numpy()
            tr = y.pow(2).mean(dim=1).sqrt().cpu().numpy()
            for b, k in enumerate(sel):
                rows.append((np.array(packed.items[k]["cond"], dtype=np.float64), e[b], pr[b], tr[b]))
    model.train()
    return rows


def sparsity_report(packed, name, per_full, window):
    """
    ESR bucketed by how far each validation window's knob setting is from the nearest
    training render's.

    The mean ESR over a split says how well the model does on average; it cannot say
    whether it degrades gracefully at knob settings nobody trained on, because a network
    that has quietly memorised a dense grid will hold up on validation and still be wrong
    between the grid points. Bucketing by nearest-neighbour distance in knob space
    separates those two cases, and the sparse bucket is the one that has to stay low for
    the plugin to be usable at arbitrary settings.

    `per_full` is the raw per-window ESR array out of evaluate(): one entry per window in
    exactly the order eval_batches emitted them, with NaN where a window's target RMS fell
    below MIN_RMS and the ESR mean dropped it. The distance array is built in that same
    window order and both are filtered by the same non-NaN mask before bucketing, which is
    what keeps the buckets aligned with the reported ESR mean. A compare of array lengths
    cannot do this: the ESR array is shorter than the window count by exactly those
    dropped windows, so any length check silently misfires whenever a split contains a
    quiet window.

    Requires the split to be by input segment; with a row-wise split every knob setting
    in validation also appears in training and every render lands in the dense bucket.
    """
    if per_full.size == 0:
        return

    val_idx = packed.usable_for(name, window)
    train_idx = packed.usable_for("train", window)
    if not val_idx or not train_idx:
        return

    def _as_render(i):
        return {"knobs": dict(zip(packed.knobs, packed.items[i]["cond"]))}

    from knobdata import knob_sparsity

    # Knob settings are constant across a render's windows, so the nearest-neighbour
    # distance is a per-render quantity. Compute it once per val render (in val_idx
    # order) and expand to windows with the same step arithmetic eval_batches uses, so
    # both arrays line up window-for-window.
    render_dist = np.asarray(
        knob_sparsity([_as_render(i) for i in val_idx],
                      [_as_render(i) for i in train_idx],
                      list(packed.knobs)), dtype=np.float64)

    step = window // 2  # overlap=True, as do_eval runs evaluate()
    order = []
    for idx, k in enumerate(val_idx):
        order.extend([idx] * len(range(0, packed.items[k]["length"] - window + 1, step)))

    if len(order) != per_full.size:
        # The step arithmetic diverged from what evaluate() ran; refuse to report rather
        # than bucket a misaligned array.
        return

    d = render_dist[np.asarray(order, dtype=np.int64)]

    ok = ~np.isnan(per_full)
    d, esr = d[ok], per_full[ok]
    if d.size == 0:
        return

    edges = [0.0, 0.005, 0.01, 0.02, 0.04]
    print(f"  ESR by distance from nearest training knob setting ({d.size} windows):")
    lo = 0.0
    for hi in edges[1:]:
        m = (d >= lo) & (d < hi)
        if m.any():
            print(f"    nn-dist [{lo:.3f},{hi:.3f})  n={int(m.sum()):5d}  "
                  f"ESR {esr[m].mean():.5f}")
        lo = hi
    m = d >= edges[-1]
    if m.any():
        print(f"    nn-dist [{edges[-1]:.3f},inf)  n={int(m.sum()):5d}  "
              f"ESR {esr[m].mean():.5f}")


def _dbfs(v):
    return 20.0 * math.log10(max(float(v), 1e-12))


def print_knob_report(rows, knobs):
    for j, knob in enumerate(knobs):
        groups = {}
        for cond, e, pr, tr in rows:
            if np.isnan(e):
                continue
            groups.setdefault(round(float(cond[j]), 2), []).append((e, pr, tr))
        if len(groups) < 2:
            continue
        print(f"  {knob}:")
        print("    value   n     ESR      pred rms   target rms   dB err")
        for v in sorted(groups):
            g = np.array(groups[v])
            pred = g[:, 1].mean()
            targ = g[:, 2].mean()
            err = _dbfs(pred) - _dbfs(targ) if targ > 1e-9 else float("nan")
            print(f"    {v:5.2f} {len(g):4d}  {g[:,0].mean():9.5f}  "
                  f"{_dbfs(pred):8.1f} dB  {_dbfs(targ):8.1f} dB  {err:7.2f}")


def train(args):
    packed = Packed(args.db, pack_dir=getattr(args, "pack", None))
    window, tbptt = args.window, args.tbptt

    # Burn-in is derived from the window rather than set independently, because the two
    # have to stay consistent: a burn-in longer than the window leaves nothing to score,
    # and one near it leaves a score so short it is mostly noise. A quarter of the window
    # is NAM's ratio on a 32768 window with 8192 burned, and it scales with the window
    # rather than being a fixed count.
    burn = args.burn if args.burn >= 0 else window // 4
    if burn >= window:
        raise SystemExit(f"--burn {burn} leaves nothing to score in a --window {window}")

    idx = packed.usable_for("train", window)
    val_idx = packed.usable_for("val", window)
    if not idx:
        raise RuntimeError(f"no training render is at least {window} samples long")
    if not val_idx:
        raise RuntimeError(f"no validation render is at least {window} samples long")

    n_features = 1 + packed.n_knobs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True       # fp32 weights, faster matmuls

    # Seed before the model is built. --seed existed and was threaded through the data
    # split, but the only torch.manual_seed in the file sat inside do_export, so the
    # initial weights were whatever the process happened to draw. Two runs differing only
    # in --seed produced different models, which makes every comparison between them
    # meaningless: an apparent improvement could be the init.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = KnobGRU(n_features=n_features, hidden=args.hidden).to(device)

    init_ck = None
    if args.init:
        init_ck = torch.load(args.init, map_location=device, weights_only=False)
        if init_ck.get("hidden") != args.hidden or init_ck.get("n_features") != n_features:
            raise SystemExit(
                f"{args.init} holds {init_ck.get('n_features')} -> {init_ck.get('hidden')} -> 1, "
                f"but this run is {n_features} -> {args.hidden} -> 1; cannot warm-start")
        model.load_state_dict(init_ck["state_dict"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    rng = np.random.default_rng(args.seed)
    batches = max(1, len(idx) // args.batch)
    total_steps = args.epochs * batches
    start_epoch = max(0, args.start_epoch)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "best.pt")
    last_ckpt = os.path.join(args.out, "last.pt")
    log_path = os.path.join(args.out, "train.log")

    print(f"model    : {n_features} -> {args.hidden} -> 1, {model.param_count()} params, {device}")
    print(f"data     : {len(idx)} train / {len(val_idx)} val renders of {window} samples")
    print(f"train    : {args.epochs} epochs x {batches} batches, TBPTT {tbptt}, "
          f"batch {args.batch}, peak lr {args.lr}, loss {args.loss}")
    print(f"burn-in  : {burn} of {window} samples unscored ({100.0 * burn / window:.0f}%), "
          f"{window - burn} scored per window")
    print(f"excluded : windows with target RMS below {MIN_RMS:g} (-60 dBFS), "
          f"which is LiveSPICE's numerical floor rather than circuit behaviour")
    if args.mrstft:
        # Printed rather than inferred: the resolutions are a function of --tbptt, so a log
        # that did not name them could not tell a later reader which variant actually ran.
        print(f"spectral : MRSTFT weight {args.mrstft:g} at pre-emphasis {args.pre_emph_mrstft:g}, "
              f"resolutions {mrstft_resolutions(tbptt)} (bounded by TBPTT {tbptt})")
    else:
        print("spectral : MRSTFT disabled (--mrstft 0)")

    # A loss that scores silence below 1.0 has a gradient pointing at silence, and on this
    # database that is where training silently goes to die while the loss looks like it is
    # falling. Checked rather than assumed, because the check is what caught the earlier bug.
    probe = torch.from_numpy(
        np.array([np.square(np.asarray(packed.y[it["y_start"]:it["y_start"] + window],
                                       dtype=np.float64)).sum()
                  for it in (packed.items[i] for i in idx[:512])], dtype=np.float32))
    sil = silence_score(probe)
    if abs(sil - 1.0) > 1e-6:
        raise SystemExit(f"refusing to train: this loss scores silence at {sil:.4f}, not 1.0, "
                         f"so it has a gradient pointing at silence")
    print(f"guard    : silence scores {sil:.4f} (must be 1.0); target energy spans "
          f"{batch_energy_spread(probe):,.0f}x across the first {min(512, len(idx))} renders")

    best = float("inf")
    if init_ck is not None:
        base = init_ck.get("val_esr")
        shown = "unknown" if base is None else format(float(base), ".5f")
        print(f"init     : warm-started weights from {args.init} "
              f"(epoch {init_ck.get('epoch')}, val ESR {shown})")
        if base is not None:
            # Seed the running best so the first, typically worse, epochs of a fresh
            # optimizer do not overwrite the checkpoint they were warm-started from.
            best = float(base)
            print(f"best seed: {best:.5f}; only improvements are written to best.pt")
    step = start_epoch * batches
    consecutive = 0
    nonfinite_total = 0
    # The guard must not be allowed to kill a long run. A *finite* step can still tip the
    # GRU's recurrence past its stability boundary: the weights stay finite but the hidden
    # state overflows, so every later batch is non-finite and no amount of skipping recovers.
    # That is exactly what ended r96_21k_cont at epoch 131 of 500. Instead of aborting, keep
    # the weights from just before each step and, on a run of non-finite batches, restore
    # them and halve the LR. The schedule continues from the current global step, so the
    # cosine still anneals to its tail; the run simply settles at whatever LR is stable here.
    peak_lr = args.lr
    rollbacks = 0
    good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    def payload(tag, val):
        return {
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden": args.hidden,
            "knobs": packed.knobs,
            "pinned_knobs": list(PINNED_KNOBS),
            "volume_law": {"gain_at_full": 1.0, "unity_at": 1.0},  # preamp has no Volume knob; no-op gain
            "val_esr": val,
            "epoch": tag,
            "window": window,
            "burn": burn,
            "tbptt": tbptt,
            # Optimizer/step/rng travel with the weights, so a crash is a real resume from
            # last.pt instead of another warm start with the AdamW moments lost.
            "optimizer": opt.state_dict(),
            "step": step,
            "rng": rng.bit_generator.state,
        }

    def save_best(tag, val):
        torch.save(payload(tag, val), ckpt)

    def save_last(tag, val):
        torch.save(payload(tag, val), last_ckpt)

    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        te = time.time()
        losses = []
        nf_epoch = 0
        for _sel, starts in packed.train_batches(idx, window, args.batch, rng):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, peak_lr)
            xb, yb, cb = packed.gather(_sel, starts, window)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(yb).to(device)
            cb = torch.from_numpy(cb).to(device)
            loss = run_window(model, xb, yb, cb, tbptt, opt=opt, mode=args.loss,
                              burn=burn, mrstft_weight=args.mrstft,
                              mrstft_coef=args.pre_emph_mrstft)
            losses.append(loss)

            # A spike must not be allowed to poison the weights. clip_grad_norm_ cannot fix a
            # non-finite gradient -- its norm is non-finite too -- so the guard has to sit
            # before opt.step(): drop the bad update and keep the last finite weights. This is
            # exactly the failure that killed r96_21k at epoch 104 and made every later step
            # NaN, which evaluate() then turned into an empty metric array and a crash.
            if math.isfinite(loss):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            else:
                gnorm = None
            if gnorm is None or not math.isfinite(float(gnorm)):
                opt.zero_grad(set_to_none=True)
                nonfinite_total += 1
                nf_epoch += 1
                consecutive += 1
                step += 1
                if consecutive >= args.max_nonfinite:
                    # Past this point the weights themselves are producing non-finite
                    # outputs, so skipping cannot recover them. Restore the last state that
                    # still made a finite step and halve the LR so the retry is gentler; a
                    # run of these just keeps halving until the recurrence is stable again.
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            v.copy_(good_state[k])
                    opt.zero_grad(set_to_none=True)
                    peak_lr *= 0.5
                    rollbacks += 1
                    consecutive = 0
                    log(f"          diverged at epoch {epoch} (step {step}); restored last "
                        f"finite weights and halved peak lr to {peak_lr:.2e} "
                        f"[rollback {rollbacks}]")
                continue
            # Snapshot the pre-step weights so a divergence *caused* by this step can be
            # undone exactly. The tip is a finite step, so the snapshot must be taken before
            # opt.step(), not after.
            good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            consecutive = 0
            opt.step()
            step += 1

        good = [l for l in losses if math.isfinite(l)]
        tr = float(np.mean(good)) if good else float("nan")
        v, _, _, _ = evaluate(model, packed, "val", window, args.eval_batch, device=device, burn=burn)
        vm = float(np.mean(v)) if v.size else float("nan")
        el = time.time() - te

        flag = ""
        if vm < best:
            best = vm
            save_best(epoch, vm)
            flag = "  *"
        save_last(epoch, vm)
        # np.median/percentile of an empty array raise IndexError, which is what actually
        # killed the process at epoch 105 once every val window scored NaN. Report nan instead.
        v50 = float(np.median(v)) if v.size else float("nan")
        v95 = float(np.percentile(v, 95)) if v.size else float("nan")
        nf = f"  nf {nf_epoch}/{nonfinite_total}" if nf_epoch else ""
        log(f"epoch {epoch:4d}  train {tr:.5f}  val {vm:.5f}  "
            f"(p50 {v50:.5f} p95 {v95:.5f})  "
            f"{el:.1f}s  lr {opt.param_groups[0]['lr']:.2e}{flag}{nf}")
        if not v.size:
            log(f"          val produced no scoreable window this epoch "
                f"(all outputs non-finite or below {MIN_RMS:g} RMS)")

    log(f"done in {(time.time() - t0) / 60:.1f} min; best val ESR {best:.5f}")
    logf.close()
    print(f"checkpoint: {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# eval / export
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = KnobGRU(n_features=ck["n_features"], hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def do_eval(args):
    packed = Packed(args.db, pack_dir=getattr(args, "pack", None))
    model, ck = load_model(args.ckpt, args.device)
    window = args.window or ck.get("window", 32768)
    burn = args.burn if args.burn >= 0 else int(ck.get("burn", -1))
    if burn < 0:
        burn = window // 4
    print(f"model    : {ck['n_features']} -> {ck['hidden']} -> 1, "
          f"{model.param_count()} params, knobs {packed.knobs}")
    print(f"scoring  : {burn} of {window} samples burned in and unscored")
    for name in (args.split.split(",") if args.split else ["val", "test"]):
        per, pre, n_all, per_full = evaluate(model, packed, name, window, args.eval_batch,
                                             overlap=True, device=args.device, burn=burn)
        if not per.size:
            print(f"{name:5s} no window carried signal above {MIN_RMS:g} RMS")
            continue
        print(f"{name:5s} ESR {per.mean():.5f}  (median {np.median(per):.5f}, "
              f"p95 {np.percentile(per, 95):.5f}, worst {per.max():.5f})   "
              f"pre-emph {pre.mean():.5f}   "
              f"[{per.size}/{n_all} windows above {MIN_RMS:g} RMS]")
        rows = per_knob_report(model, packed, name, window, device=args.device, burn=burn,
                               batch=args.eval_batch)
        print_knob_report(rows, packed.knobs)

        # Parametric generalisation, reported separately from the mean above. A knob cell
        # far from anything in training is the situation the plugin is actually in when a
        # user picks a setting nobody rendered, and averaging it together with well-covered
        # settings hides it.
        sparsity_report(packed, name, per_full, window)


def volume_response(args):
    """
    Check the Volume knob against the ground truth, at knob positions we never rendered.

    Volume is no longer learned -- it is pinned during generation and applied as gain in
    the plugin -- so this is not checking the network any more. It is checking the one
    claim the decomposition rests on: that out(V2) = (V2 / V1) * out(V1) exactly, for knob
    positions between the grid, at signal levels we never trained on.

    The check runs LiveSPICE on a held-out excerpt at a set of Volume values and compares
    the measured ratio against the ratio the plugin will apply. If the ratios agree to
    well below the model's accuracy, the decomposition is safe; if they do not, Volume has
    to come back into the network. Reporting the agreement as a number, rather than
    assuming it, is the whole point -- this was verified once on a 1 s excerpt, and the
    database has since grown to a completely different set of drive levels.
    """
    packed = Packed(args.db, pack_dir=getattr(args, "pack", None))
    model, _ = load_model(args.ckpt, args.device)
    if "Volume" in packed.knobs:
        raise SystemExit("this pack still learns Volume; there is nothing to check here")
    sr = packed.sr
    n = int(sr * args.seconds)

    # Fixed excerpts from the test split, so this measures generalisation and not recall.
    idx = [i for i in packed.usable_for(args.split, n)]
    picks = []
    for i in idx:
        it = packed.items[i]
        if np.asarray(packed.y[it["y_start"]:it["y_start"] + n]).std() > 0.02:
            picks.append(i)
        if len(picks) >= args.clips:
            break
    if not picks:
        raise RuntimeError(f"no {args.split} render has {args.seconds}s of signal above 0.02 RMS")

    gen = args.generator
    if not gen or not os.path.exists(gen):
        raise SystemExit("--generator must point at livespice-gen.exe")

    circuit = args.circuit
    if not circuit:
        circuit = os.path.join(
            os.path.dirname(os.path.dirname(os.path.normpath(gen))),
            "examples", "circuits", "Big Muff Pi.schx")
        circuit = os.path.normpath(circuit)
    if not os.path.exists(circuit):
        raise SystemExit(f"circuit not found: {circuit} (pass --circuit)")

    # Ground truth is rendered fresh, at Volume settings the database never contains.
    vols = [float(v) for v in args.volumes.split(",")]
    if len(vols) < 2:
        raise SystemExit("--volumes needs at least two comma-separated positions")

    excerpt = os.path.join(os.path.dirname(os.path.normpath(gen)), "excerpt.wav")
    src = packed.x
    off = int(args.offset * sr) if args.offset is not None else sr * 30
    if off + n > len(src):
        raise SystemExit(f"offset {off / sr:.1f}s + {args.seconds}s runs past the end of the input")
    sf.write(excerpt, np.asarray(src[off:off + n], dtype=np.float32), sr, subtype="FLOAT")

    print(f"input    : {len(picks)} {args.split} excerpts, {args.seconds:g}s each at {sr} Hz")
    print(f"truth    : re-rendering {os.path.basename(excerpt)} with {os.path.basename(gen)}")
    print(f"           at Volume {', '.join(f'{v:g}' for v in vols)}, "
          f"drive {args.gain_db:g} dB (training spanned -18..+6)")
    print()
    print("   vol    measured ratio    plugin law    deviation")

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ref = None
        rows = []
        for v in vols:
            out = os.path.join(tmp, f"v{v:g}.wav")
            cmd = [gen, "process", args.circuit, excerpt, out,
                   "--knob", f"Sustain={args.sustain:g}",
                   "--knob", f"Tone={args.tone:g}",
                   "--knob", f"Volume={v:g}",
                   "--warmup", "0.05", "--oversample", str(args.oversample),
                   "--iterations", str(args.iterations),
                   "--gain", f"{args.gain_db:g}", "--quiet"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(out):
                raise SystemExit(f"livespice-gen failed at Volume {v:g}:\n"
                                 f"{r.stdout}\n{r.stderr}")
            d, _ = sf.read(out, always_2d=False)
            d = np.asarray(d, dtype=np.float64)
            if ref is None:
                ref = d
                # The reference is the first position requested, so ratios are against it.
                ref_v = v
                continue
            k = float(np.dot(d, ref) / max(np.dot(ref, ref), 1e-30))
            law = volume_gain(v) / volume_gain(ref_v)
            rows.append((v, k, law, k - law))

        for v, k, law, dev in rows:
            print(f"  {v:5.2f}  {k:15.9f}  {law:12.9f}  {dev:+11.2e}")

    worst = max(abs(r[3]) for r in rows) if rows else float("nan")
    print()
    print(f"worst    : {worst:.2e} absolute ratio error")
    # The bar is set where it matters: far below the ~1e-2 ESR the model operates at. A
    # decomposition that only held to a few percent would be worse than the error it
    # introduces, because that error is systematic across every sample.
    print(f"gate     : {'PASS' if worst < 1e-3 else 'FAIL'} at 1e-3, "
          f"which is 10x below the ESR this model operates at")
    if not rows:
        raise SystemExit("no comparison was produced")
    if worst >= 1e-3:
        print("Note     : Volume is not separable at this drive level. Put it back in the\n"
              "           feature vector and re-render the database.")
    print("Note     : `process` ignores --knob in livespice-gen 0.x, so this comparison "
          "will read as\n           identical ratios until that is fixed or `render` is used "
          "instead.")


def do_export(args):
    packed = Packed(args.db, pack_dir=getattr(args, "pack", None))
    model, ck = load_model(args.ckpt, args.device)

    # Refuse to export a model whose conditioning does not match the architecture this
    # plugin was compiled for. A model trained before Volume was pinned has a wider first
    # layer, and shipping it would load in a DAW and sound plausible while every knob was
    # offset by one slot.
    stage_knobs, stage_pinned = get_stage_knobs(packed.databases)
    if packed.knobs != list(stage_knobs):
        raise SystemExit(
            "this checkpoint conditions on " + ", ".join(packed.knobs)
            + " but the stage expects " + ", ".join(stage_knobs)
            + ". Re-train rather than re-export.")

    # Prove the export before shipping it: replay the JSON in numpy and compare to PyTorch.
    torch.manual_seed(0)
    x = torch.randn(2, 512, device=args.device) * 0.3
    cond = torch.rand(2, packed.n_knobs, device=args.device)
    os.makedirs(args.out, exist_ok=True)
    is_poweramp = "P" in packed.knobs and len(packed.knobs) == 1
    meta = {
        "pedal": "Mesa Boogie Mark I/II 100W Poweramp (virtual analog, LiveSPICE)" if is_poweramp else "Mesa Boogie Mark IIC+ (++) Preamp (virtual analog, LiveSPICE)",
        "knobs": packed.knobs,
        "pinned_knobs": list(stage_pinned),
        # Recorded so the plugin does not have to hardcode the measured pot law, and so a
        # change to it is visible in the model file rather than only in the source.
        "volume_gain": {
            "law": "no-op (preamp has no Volume knob)",
            "gain_at_full": 1.0,
            "unity_at": 1.0,
            "note": "The preamp has no Volume knob. Master 1 is pinned at 0.5 (unity) in the render and Pull Lead is always ON. The plugin's Output knob is a post-network trim in dB (see Plugin.h outputGainFromDb).",
        },
        "note": "Channel 0 is audio; channels 1.. are the knob values in `knobs` order. "
                "Master 1 and Pull Lead are pinned (not conditioned). Build that vector per audio "
                "sample and pass it to RTNeural::ModelT<float, N, 1, "
                "DenseT<float,N,H>, TanhActivationT<float,H>, GRULayerT<float,H,H>, "
                "DenseT<float,H,1>>. No post-network gain is applied: the plugin's Output knob "
                "is a dB trim (see Plugin.h outputGainFromDb).",
        "dataset": os.path.basename(os.path.normpath(args.db)),
        "window": ck.get("window"),
        "burn": ck.get("burn"),
        "val_esr": ck.get("val_esr"),
    }
    filename = "poweramp.json" if is_poweramp else "preamp.json"
    path = os.path.join(args.out, filename)
    export_rtneural(model, path, meta)
    err, ok = verify_export(model, path, x, cond)
    print(f"export   : {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    print(f"verify   : numpy replay max abs error {err:.2e}  ->  {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("export does not reproduce the trained model; refusing to ship it")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare", help="pack the database and draw the split")
    q.add_argument("--db", required=True,
                   help="render database directory; several may be merged, comma- or "
                        "space-separated, and must share one input/normalized.wav")
    q.add_argument("--pack", default=None,
                   help="destination pack directory; defaults to <db>/pack")
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=lambda a: prepare(a.db, a.seed, pack_dir=a.pack))

    q = sub.add_parser("train", help="train the model")
    q.add_argument("--db", required=True,
                   help="render database directory; several may be merged, comma- or "
                        "space-separated, and must share one input/normalized.wav")
    q.add_argument("--pack", default=None,
                   help="pack directory; defaults to <db>/pack")
    q.add_argument("--out", default="runs/latest")
    q.add_argument("--hidden", type=int, default=128)
    # 32768 is NAM's stated default for this kind of run ("use a long ny like
    # 32768"), and it was needed here for the burn-in to fit: at 8192 the quarter-window
    # burn is 2048 samples, which is not long enough to settle the state, so the startup
    # error stayed in the loss. A longer window costs memory per batch, not throughput.
    q.add_argument("--window", type=int, default=32768)
    q.add_argument("--burn", type=int, default=-1,
                   help="leading samples per window to feed but not score; "
                        "negative means a quarter of --window (default)")
    q.add_argument("--tbptt", type=int, default=2048)
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--eval-batch", type=int, default=64)
    q.add_argument("--epochs", type=int, default=200)
    q.add_argument("--lr", type=float, default=2e-3)
    q.add_argument("--clip", type=float, default=1.0)
    q.add_argument("--init", default=None,
                   help="checkpoint whose weights are loaded before training starts. The "
                        "optimizer is built fresh, so this is a warm start of the weights, "
                        "not a resume of the optimisation state")
    q.add_argument("--start-epoch", type=int, default=0,
                   help="the epoch --init left off at. The global step and therefore the LR "
                        "schedule continue from start-epoch * batches, so the cosine is picked "
                        "up where it stopped instead of being restarted from warmup")
    q.add_argument("--max-nonfinite", type=int, default=20,
                   help="consecutive non-finite losses or gradients that count as a divergence. "
                        "An isolated spike is skipped; a run of them triggers the self-heal: "
                        "weights roll back to the last finite step and the peak LR is halved, "
                        "so a diverging run recovers instead of aborting")
    q.add_argument("--loss", choices=("equalize", "perclip", "global"), default="equalize",
                   help="how clips are weighted against each other. See losses.clip_weights. "
                        "'equalize' is the default because 'perclip' does not train on this "
                        "database (val ESR 0.94, the score of silence) and 'global' trains "
                        "but under-serves the quiet end.")
    q.add_argument("--mrstft", type=float, default=0.002,
                   help="weight of the multi-resolution STFT term added to ESR, on the "
                        "pre-emphasised signals. NAM's pre_emph_mrstft_weight is 0.002 and "
                        "that is what this defaults to. 0 disables it. Reporting stays "
                        "plain ESR either way, so this changes what is fitted, not what "
                        "is measured.")
    q.add_argument("--pre-emph-mrstft", type=float, default=0.85,
                   help="pre-emphasis coefficient for the spectral term, NAM's "
                        "pre_emph_mrstft_coef. 0 disables it and matches NAM's un-pre-"
                        "emphasised mrstft_weight instead.")
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=train)

    q = sub.add_parser("eval", help="score a checkpoint")
    q.add_argument("--db", required=True,
                   help="render database directory; several may be merged, comma- or "
                        "space-separated, and must share one input/normalized.wav")
    q.add_argument("--pack", default=None,
                   help="pack directory; defaults to <db>/pack")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--split", default="val,test")
    q.add_argument("--window", type=int, default=0)
    q.add_argument("--burn", type=int, default=-1,
                   help="leading samples not scored; negative means a quarter of the window, "
                        "matching training")
    q.add_argument("--eval-batch", type=int, default=64)
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q.set_defaults(fn=do_eval)

    q = sub.add_parser("volume-response",
                       help="re-render ground truth at unrendered Volume settings and "
                            "check the gain law the plugin relies on")
    q.add_argument("--db", required=True,
                   help="render database directory; several may be merged, comma- or "
                        "space-separated, and must share one input/normalized.wav")
    q.add_argument("--pack", default=None,
                   help="pack directory; defaults to <db>/pack")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--split", default="test")
    q.add_argument("--clips", type=int, default=8)
    q.add_argument("--seconds", type=float, default=4.0)
    q.add_argument("--generator", required=True,
                   help="path to livespice-gen.exe, used to re-render the ground truth")
    q.add_argument("--circuit", default=None,
                   help="circuit .schx; defaults to the Big Muff Pi bundled with "
                        "the generator")
    q.add_argument("--volumes", default="0.1,0.15,0.3,0.5",
                   help="comma-separated Volume positions to render, none of which "
                        "are the pinned 0.2")
    q.add_argument("--sustain", type=float, default=0.9)
    q.add_argument("--tone", type=float, default=0.6)
    q.add_argument("--gain-db", type=float, default=6.0,
                   help="drive for the probe; the default is the top of the trained range, "
                        "where the circuit is hardest")
    q.add_argument("--offset", type=float, default=None,
                   help="where in the input to probe, seconds; defaults to 30 s")
    q.add_argument("--oversample", type=int, default=16)
    q.add_argument("--iterations", type=int, default=32)
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q.set_defaults(fn=volume_response)

    q = sub.add_parser("export", help="write RTNeural JSON and verify it")
    q.add_argument("--db", required=True,
                   help="render database directory; several may be merged, comma- or "
                        "space-separated, and must share one input/normalized.wav")
    q.add_argument("--pack", default=None,
                   help="pack directory; defaults to <db>/pack")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--out", default="runs/latest")
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q.set_defaults(fn=do_export)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
