"""
What level does the model actually produce, and over what input range is it trustworthy?

Two things need answering before the plugin can ship:

  1. The plugin's V/S/T defaults are currently arbitrary numbers. They have to come
     from measured behaviour, not taste.

  2. "The plugin is quiet" may not be a bug. The model reproduces the pedal's true
     level and has no output make-up gain, so its output level is set almost entirely
     by how hot the input is. If the model is only valid over a narrow input band, the
     honest answer is to document that band and let Input Trim cover the rest.

The training database swept input gain over -18..+6 dB log-uniform, so the model was
never shown inputs outside roughly -40..-16 dBFS RMS. Anything outside that is
extrapolation and is reported separately rather than blended in.

Run:  python levelmap.py [--ckpt runs/e_h128_t2048/best.pt] [--ckpt ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Packed, load_model                       # noqa: E402

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'bigmuff-v1')

MIN_RMS = 1e-3          # -60 dBFS, the same gate the loss uses


def db(v):
    return 20.0 * np.log10(max(float(v), 1e-12))


def run(model, x, cond, device, flush=8192):
    import torch
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    ct = torch.from_numpy(cond.astype(np.float32)).unsqueeze(0).to(device)
    zeros = torch.zeros(1, flush, device=device)
    out = []
    with torch.no_grad():
        _, st = model(zeros, ct)
        for s in range(0, xt.shape[1], 16384):
            y, st = model(xt[:, s:s + 16384], ct, st)
            out.append(y.squeeze(-1))
    return torch.cat(out, 1)[0].cpu().numpy().astype(np.float64)


def esr(ref, test):
    """Masked ESR, matching the trainer: silent targets carry no information."""
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.ndim == 1:
        ref, test = ref[None, :], test[None, :]
    # One mask per window, flattened so it indexes the per-window sums directly.
    sel = (np.sqrt((ref ** 2).mean(axis=-1)) >= MIN_RMS)
    if not sel.any():
        return None
    num = ((ref - test) ** 2).sum(axis=-1)[sel]
    den = (ref ** 2).sum(axis=-1)[sel]
    return float((num / den).mean())


def probe_material(pk, seconds=2.0):
    """A real audio excerpt at a known loudness, taken from the sweep input."""
    want = int(seconds * pk.sr)
    start = int(17.0 * pk.sr)          # inside the music region of the sweep file
    x = pk.x[start:start + want].astype(np.float64)
    return x, db(np.sqrt((x ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    ckp = args.ckpt or [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'runs', 'e_h128_t2048', 'best.pt')]

    pk = Packed(DB)
    x0, x0_db = probe_material(pk)
    print('probe material: %.1f s, RMS %.1f dBFS\n' % (len(x0) / pk.sr, x0_db))

    # Input levels spanning the trained band and well past it, so the shape of any
    # extrapolation is visible rather than guessed at.
    gains_db = [-24, -18, -12, -6, 0]
    knob_sets = [
        ('clean',        (0.60, 0.00, 0.50)),
        ('default-ish',  (0.70, 0.60, 0.50)),
        ('medium drive', (0.60, 0.60, 0.50)),
        ('full drive',   (0.50, 1.00, 0.50)),
        ('quiet drive',  (0.10, 0.60, 0.50)),
    ]

    report = {'probe_rms_dbfs': round(x0_db, 2), 'rows': []}

    for ck in ckp:
        model, meta = load_model(ck, args.device)
        name = os.path.basename(os.path.dirname(ck))
        print('=' * 78)
        print('checkpoint: %s' % name)
        print('=' * 78)
        hdr = '%-14s %8s %8s %9s %9s %7s' % ('setting', 'V/S/T', 'in dBFS', 'out dBFS', 'out pk', 'clip%')
        print(hdr)
        print('-' * len(hdr))

        for label, cond in knob_sets:
            for gd in gains_db:
                x = x0 * 10.0 ** (gd / 20.0)
                y = run(model, x, np.array(cond, dtype=np.float32), args.device)
                o_rms = db(np.sqrt((y ** 2).mean()))
                o_pk = db(np.abs(y).max())
                clip = 100.0 * float((np.abs(y) > 1.0).mean())
                print('%-14s %8s %8.1f %9.1f %9.1f %6.1f%%'
                      % (label, '/'.join('%.2f' % c for c in cond),
                         x0_db + gd, o_rms, o_pk, clip))
                report['rows'].append({
                    'ckpt': name, 'setting': label,
                    'knobs': [float(c) for c in cond],
                    'in_dbfs': round(x0_db + gd, 2),
                    'out_rms_dbfs': round(o_rms, 2),
                    'out_peak_dbfs': round(o_pk, 2),
                    'clip_pct': round(clip, 3),
                })
            print()

    # How trustworthy is the model as a function of input level? Only held-out renders,
    # bucketed by their own input RMS, which is what the database actually covered.
    print('=' * 78)
    print('accuracy vs input level (held-out test split, 4096-sample windows)')
    print('=' * 78)
    model, _ = load_model(ckp[0], args.device)
    EXCERPT = 16384                      # 0.34 s is ample to judge a level bucket
    buckets = [(-70, -55), (-55, -45), (-45, -35), (-35, -25), (-25, -10)]
    agg = {b: [] for b in buckets}
    for k in pk.usable_for('test', EXCERPT):
        it = pk.items[k]
        x = pk.x[it['in_start']:it['in_start'] + it['length']].astype(np.float64) * it['gain']
        tgt = np.asarray(pk.y[it['y_start']:it['y_start'] + it['length']], dtype=np.float64)
        # Take the first non-silent excerpt; the silent head carries no signal to judge.
        for off in range(0, max(1, len(x) - EXCERPT + 1), EXCERPT):
            xe, te = x[off:off + EXCERPT], tgt[off:off + EXCERPT]
            if np.sqrt((te ** 2).mean()) < MIN_RMS:
                continue
            i_rms = db(np.sqrt((xe ** 2).mean()))
            y = run(model, xe, pk.cond_all[k], args.device)
            e = esr(te, y)
            for b in buckets:
                if b[0] <= i_rms < b[1] and e is not None:
                    agg[b].append({
                        'in_db': i_rms,
                        'tgt_db': db(np.sqrt((te ** 2).mean())),
                        'pred_db': db(np.sqrt((y ** 2).mean())),
                        'esr': e,
                    })
            break

    print('%-14s %5s %8s %9s %9s %10s %9s'
          % ('input bucket', 'n', 'in dBFS', 'tgt dBFS', 'pred dBFS', 'lvl err', 'ESR'))
    for b in buckets:
        rows = agg[b]
        if not rows:
            print('%-14s %5d  (no renders here)' % ('%d..%d dBFS' % b, 0))
            continue
        in_db = np.array([r['in_db'] for r in rows])
        tg_db = np.array([r['tgt_db'] for r in rows])
        pr_db = np.array([r['pred_db'] for r in rows])
        es = np.array([r['esr'] for r in rows])
        tag = '' if b[0] >= -55 else '  <- extrapolated'
        print('%-14s %5d %8.1f %9.1f %9.1f %9.2f dB %9.5f%s'
              % ('%d..%d dBFS' % b, len(rows), np.median(in_db), np.median(tg_db),
                 np.median(pr_db), np.median(pr_db - tg_db), np.median(es), tag))

    # The headline question: does the model's output level track ground truth across the
    # input range, or has it collapsed to a knob-determined constant? A flat transfer
    # curve is only alarming if the targets are not also flat, so this has to be
    # compared against the targets rather than read off the curve above.
    print('\nlevel error vs ground truth by input bucket (positive = model too loud):')
    for b in buckets:
        rows = agg[b]
        if len(rows) < 3:
            continue
        d = np.array([r['pred_db'] - r['tgt_db'] for r in rows])
        print('  %-14s n=%3d  median %+.2f dB  p90 %+.2f dB  worst %+.2f dB'
              % ('%d..%d dBFS' % b, len(rows), np.median(d),
                 np.percentile(d, 90), d[np.argmax(np.abs(d))]))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'levelmap.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print('\nwrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())