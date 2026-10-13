"""
Build a listen pack: dry / target / model for held-out renders, plus a blind A/B set.

The pack answers one question by ear that no number answers well: how close is the network
to the circuit it was trained on. It is deliberately the same three signals every time --
dry (what the circuit saw), target (LiveSPICE's output), model (the network's output) -- with
ONE gain per render shared by all three, so the level relationship between them is part of
what gets judged. Normalising per file would hide level errors.

Why a script rather than the ad-hoc command that produced the v1 pack: the v1 key leaked
into MANIFEST.json, which quietly destroyed the blind test. The key is written to its own
file, outside the pack directory, so `blind/` can stay genuinely blind.

Usage:
    python training/listenpack.py --db datasets/bigmuff-v1 \
        --ckpt training/runs/e_h128_t2048/best.pt --out datasets/listen-ab-v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from losses import MIN_RMS, esr_per_clip, esr_pre, pre_emphasis, usable_mask   # noqa: E402
from train import Packed, load_model                                          # noqa: E402


# The eight renders the v1 pack used. Reusing them is the point: the listener already has a
# verdict for these exact clips, so a v2 pack on the same renders turns "it sounded close"
# into a measurement of whether it got closer. Changing the clips would throw that away.
DEFAULT_RENDERS = [
    "BigMuffPi_S18_T29_V79_t90.00-100.00s",
    "BigMuffPi_S70_T10_V80_t110.00-120.00s",
    "BigMuffPi_S90_T30_V50_t60.00-70.00s",
    "BigMuffPi_S27_T86_V37_t90.00-100.00s",
    "BigMuffPi_S90_T20_V20_t110.00-120.00s",
    "BigMuffPi_S20_T30_V20_t120.00-130.00s",
    "BigMuffPi_S40_T60_V20_t20.00-30.00s",
    "BigMuffPi_S10_T100_V70_t110.00-120.00s",
]

# The plugin primes the recurrent state with this many silent samples in prepareToPlay
# (Plugin.h flushSamples). Doing the same here means the pack auditions what the plugin will
# actually play, rather than a state the plugin never uses.
FLUSH_SAMPLES = 8192
CHUNK = 8192


def dbfs(v: float) -> float:
    return 20.0 * math.log10(max(float(v), 1e-12))


def run_model(model, x: np.ndarray, cond: np.ndarray, device: str) -> np.ndarray:
    """Free-run the network over `x`, carrying GRU state, after a silence flush."""
    ct = torch.from_numpy(cond).unsqueeze(0).to(device)
    xt = torch.from_numpy(x).float().unsqueeze(0).to(device)
    out = []
    with torch.no_grad():
        state = None
        if FLUSH_SAMPLES:
            _, state = model(torch.zeros(1, FLUSH_SAMPLES, device=device), ct)
        for s in range(0, xt.shape[1], CHUNK):
            y, state = model(xt[:, s:s + CHUNK], ct, state)
            out.append(y.squeeze(-1))
    return torch.cat(out, 1)[0].cpu().numpy().astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key-out", default=None,
                   help="where to write the blind key; defaults to <out>.key.json "
                        "(outside the pack directory on purpose)")
    p.add_argument("--renders", default=None,
                   help="comma-separated render ids; default is the v1 set")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--peak-ceiling", type=float, default=-1.0,
                   help="dBFS the target peak is scaled to, shared by dry/target/model")
    p.add_argument("--seed", type=int, default=1, help="controls the A/B assignment")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ids = args.renders.split(",") if args.renders else DEFAULT_RENDERS
    key_out = args.key_out or (os.path.normpath(args.out) + ".key.json")

    packed = Packed(args.db)
    model, ck = load_model(args.ckpt, args.device)
    model.eval()

    by_id = {it["id"]: k for k, it in enumerate(packed.items)}
    missing = [r for r in ids if r not in by_id]
    if missing:
        raise SystemExit("not in the packed database: " + ", ".join(missing))

    n = int(round(packed.sr * args.seconds))
    lab = os.path.join(args.out, "labeled")
    bld = os.path.join(args.out, "blind")
    os.makedirs(lab, exist_ok=True)
    os.makedirs(bld, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    ceiling = 10.0 ** (args.peak_ceiling / 20.0)
    rows = []
    key = {}

    for i, rid in enumerate(ids, start=1):
        k = by_id[rid]
        it = packed.items[k]
        if it["length"] < n:
            raise SystemExit(f"{rid}: only {it['length'] / packed.sr:.1f}s, need {args.seconds:g}s")

        x, y, cond = packed.gather([k], [0], n)
        x = x[0].astype(np.float64)
        y = y[0].astype(np.float64)
        m = run_model(model, x, cond[0], args.device)

        # One gain for all three, set by the target peak. Scaling by the target rather than by
        # each file keeps a level error between model and target audible instead of erased.
        g = ceiling / max(np.abs(y).max(), 1e-12)
        xs, ys, ms = (x * g, y * g, m * g)

        stem = "%02d_%s" % (i, rid.replace("BigMuffPi_", ""))
        wavfile.write(os.path.join(lab, stem + "_dry.wav"), packed.sr, xs.astype(np.float32))
        wavfile.write(os.path.join(lab, stem + "_target.wav"), packed.sr, ys.astype(np.float32))
        wavfile.write(os.path.join(lab, stem + "_model.wav"), packed.sr, ms.astype(np.float32))

        # Blind pair: same shared gain, so switching between A and B does not jump in level.
        first_is_model = bool(rng.randint(0, 2))
        a, b = (ms, ys) if first_is_model else (ys, ms)
        wavfile.write(os.path.join(bld, stem + "_A.wav"), packed.sr, a.astype(np.float32))
        wavfile.write(os.path.join(bld, stem + "_B.wav"), packed.sr, b.astype(np.float32))
        key["%02d" % i] = {"A": "model" if first_is_model else "target",
                           "B": "target" if first_is_model else "model"}

        # Same metric the training numbers use, so these ESRs are comparable with the run log.
        with torch.no_grad():
            pt = torch.from_numpy(m).float().unsqueeze(0)
            yt = torch.from_numpy(y).float().unsqueeze(0)
            if bool(usable_mask(yt)[0]):
                e = float(esr_per_clip(pt, yt)[0])
                e_pre = float(esr_pre(pt, yt))
            else:
                e = e_pre = float("nan")     # below LiveSPICE's -60 dBFS floor; not a failure

        diff = ys - ms
        corr = float(np.corrcoef(ys, ms)[0, 1])
        rows.append({
            "n": "%02d" % i,
            "id": rid,
            "knobs_VST": [float(v) for v in cond[0]],
            "split": it["split"],
            "esr": None if math.isnan(e) else round(e, 5),
            "esr_preemph": None if math.isnan(e_pre) else round(e_pre, 5),
            "correlation": round(corr, 4),
            "diff_db_below_target": round(dbfs(np.sqrt((diff ** 2).mean()))
                                          - dbfs(np.sqrt((ys ** 2).mean())), 1),
            "target_rms_dbfs": round(dbfs(np.sqrt((ys ** 2).mean())), 1),
            "model_rms_dbfs": round(dbfs(np.sqrt((ms ** 2).mean())), 1),
            "dry_rms_dbfs": round(dbfs(np.sqrt((xs ** 2).mean())), 1),
            "target_peak_dbfs": round(dbfs(np.abs(ys).max()), 1),
            "applied_gain_db": round(dbfs(g), 1),
        })
        print("%s  %-34s V/S/T %-18s ESR %-8s pre %-8s corr %.4f  diff %5.1f dB"
              % (rows[-1]["n"], rid.replace("BigMuffPi_", ""),
                 "/".join("%.2f" % v for v in cond[0]),
                 rows[-1]["esr"], rows[-1]["esr_preemph"], corr,
                 rows[-1]["diff_db_below_target"]))

    esrs = [r["esr"] for r in rows if r["esr"] is not None]
    meta = {
        "checkpoint": os.path.abspath(args.ckpt),
        "val_esr": ck.get("val_esr"),
        "hidden": ck.get("hidden"),
        "window": ck.get("window"),
        "tbptt": ck.get("tbptt"),
        "dataset": os.path.basename(os.path.normpath(args.db)),
        "split": "test (held out; never seen in training)",
        "min_rms_mask": MIN_RMS,
        "peak_ceiling_dbfs": args.peak_ceiling,
        "flush_samples": FLUSH_SAMPLES,
        "renders": rows,
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(key_out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": os.path.abspath(args.ckpt), "keys": key}, f, indent=2)

    write_readme(args.out, meta, key_out, rows)

    print()
    print("wrote     : %s" % os.path.abspath(args.out))
    print("blind key : %s   <- NOT in the pack" % os.path.abspath(key_out))
    if esrs:
        print("ESR       : mean %.5f  median %.5f  worst %.5f  (v1 pack: mean 0.0248)"
              % (float(np.mean(esrs)), float(np.median(esrs)), float(np.max(esrs))))


def write_readme(out, meta, key_out, rows) -> None:
    ranked = sorted([r for r in rows if r["esr"] is not None], key=lambda r: -r["esr"])
    lines = []
    A = lines.append
    A("# Listen A/B - Big Muff Pi neural model (%s)" % os.path.basename(os.path.normpath(meta["checkpoint"])))
    A("")
    A("Checkpoint: `%s`" % meta["checkpoint"])
    A("")
    A("Val ESR **%.5f**, hidden %s, TBPTT %s, window %s."
      % (meta["val_esr"], meta["hidden"], meta["tbptt"], meta["window"]))
    A("")
    A("All renders are from the **%s** split. The network never saw them in training." % meta["split"])
    A("")
    A("## Do the blind test first")
    A("")
    A("`blind/` - %d pairs, A and B, level-matched, so switching does not jump in loudness."
      % len(rows))
    A("")
    A("Decide which is which for each pair and write the numbers down **before** opening")
    A("`labeled/`. Once you have heard the labeled files the blind test tells us nothing.")
    A("")
    A("The key is not in this folder. It is at:")
    A("")
    A("```")
    A(key_out)
    A("```")
    A("")
    A("## Then the labeled files")
    A("")
    A("`labeled/` - same %d renders, three files each:" % len(rows))
    A("")
    A("- `_dry.wav` the guitar input the circuit saw, before the pedal")
    A("- `_target.wav` the LiveSPICE circuit output - the ground truth")
    A("- `_model.wav` the network's output")
    A("")
    A("All three share one gain per render, set so the target peaks at %.0f dBFS. Nothing is"
      % meta["peak_ceiling_dbfs"])
    A("normalised per file, because the level relationship between dry, target and model is")
    A("part of what is being tested.")
    A("")
    A("## Measured fit")
    A("")
    A("| n | V | S | T | ESR | pre-emph ESR | corr | diff sits |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A("| %s | %.2f | %.2f | %.2f | %s | %s | %.4f | %.1f dB below |"
          % (r["n"], r["knobs_VST"][0], r["knobs_VST"][1], r["knobs_VST"][2],
             "-" if r["esr"] is None else "%.4f" % r["esr"],
             "-" if r["esr_preemph"] is None else "%.4f" % r["esr_preemph"],
             r["correlation"], r["diff_db_below_target"]))
    A("")
    if ranked:
        A("Worst fit: %s (ESR %.4f). Best fit: %s (ESR %.4f)."
          % (ranked[0]["n"], ranked[0]["esr"], ranked[-1]["n"], ranked[-1]["esr"]))
        A("If you can only hear a difference on the worst-fitting ones, that is a good result.")
    A("")
    A("`pre-emph ESR` is the same fit measured after a 0.95 pre-emphasis filter, which weights")
    A("the high end. Where it is much worse than plain ESR, the residual is brightness rather")
    A("than level - and that is the confound below.")
    A("")
    A("## What a difference does and does not mean")
    A("")
    A("**If you hear a clear, character-level difference** (tone colour, fuzz texture, level)")
    A("on most pairs, the network is approximating the waveform without fully capturing the")
    A("circuit. That is expected of any model at this size.")
    A("")
    A("**If the pairs are near-indistinguishable**, that is evidence the network captured this")
    A("circuit - but it is *not* evidence the circuit is a real Big Muff. The topology was")
    A("inferred from component values and never traced from a schematic. That question is")
    A("separate, and `../listen-pack/` is where it gets tested.")
    A("")
    A("**Known confound:** LiveSPICE decimates to 48 kHz with a boxcar average rather than a")
    A("proper filter, so the targets may carry aliasing that a 48 kHz model cannot reproduce.")
    A("If the target sounds brighter or harsher than the model, that may be the renderer's")
    A("decimation rather than a model defect. These two cannot currently be separated.")
    A("")
    A("Each render is primed with %d silent samples before the audio, which is what the plugin"
      % meta["flush_samples"])
    A("does in `prepareToPlay`, so the pack auditions what the plugin will actually play.")
    A("")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
