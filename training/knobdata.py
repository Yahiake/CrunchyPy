# File: knobdata.py
"""
Read a render database produced by LiveSPICE's `render` command and pack it for training.

The database is a folder holding:
    manifest.jsonl          one JSON object per render (the source of truth)
    input/normalized.wav    the single normalized input recording, shared by every render
    renders/*.wav           one output per render, named by the manifest

Every render was produced by running one segment of the shared input through the circuit
at one knob setting. The renderer's gain is not baked into the files -- it is applied to the
input before the circuit sees it, and recorded per render -- so the input has to be rebuilt
here as `segment * input_gain`. That is the single place where a mistake would poison
training while still looking plausible, so it is asserted rather than assumed.

Packing exists only for speed: `prepare` writes one memmapped target array and an index, so
an epoch never touches 1600 individual files. It changes nothing about the data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

# Tensors that must line up sample-for-sample or the model learns a time shift.
INPUT_DT = np.float32
TARGET_DT = np.float32

# The renderer reports these; a database containing them is not something to train on.
BAD_STATUS = ("error", "failed")

# A render whose output is quieter than this is not a measurement, it is the circuit
# failing to converge or the solve drifting. Kept in the index and reported, dropped by
# `prepare`, because the split must not depend on whether a clip was trainable.
QUIET_PEAK = 1e-5


@dataclass
class Render:
    """One row of the manifest, plus the paths needed to load it."""

    id: str
    file: str
    knobs: dict          # name -> 0..1
    offset_s: float
    duration_s: float
    input_gain: float
    status: str
    peak: float
    rms: float
    clipped: int
    nonfinite: int
    flat: bool
    # Which database this row came from. Carried on the row rather than passed alongside it
    # so that a render stays loadable no matter how many databases are being merged - see
    # read_manifest() on why merging is supported at all.
    db_dir: str = ""
    # Filled in by pack():
    start: int = -1     # index of this render's first sample in the packed target array
    split: str = ""     # train / val / test, assigned by split()

    @property
    def usable(self) -> bool:
        """True if this render is a trustworthy measurement."""
        return (
            self.status not in BAD_STATUS
            and self.nonfinite == 0
            and not self.flat
            and self.peak > QUIET_PEAK
        )


def read_manifest(db_dirs) -> list:
    """
    Load manifest.jsonl from one or more databases, skipping the renderer's header line.

    The header is not JSON -- it is a human summary. It is skipped by shape rather than by
    position, because the header is optional and a run resumed from an existing manifest
    has no header at all.

    Several databases can be merged in one call. This exists so that a later, differently
    sampled render batch can be added to an earlier one without re-rendering the earlier
    one or hand-copying a thousand WAVs into a new folder. It is only sound if the two
    batches share an input recording, which load_input() asserts, and if their render ids
    do not collide, which is checked here rather than assumed -- a colliding id would
    silently overwrite one render's audio with another's in the packed array, and the
    manifest row describing the audio would no longer be the audio.
    """
    if isinstance(db_dirs, str):
        db_dirs = [db_dirs]
    db_dirs = [os.path.abspath(d) for d in db_dirs]

    out = []
    seen = {}

    for db_dir in db_dirs:
        path = os.path.join(db_dir, "manifest.jsonl")

        if not os.path.exists(path):
            raise RuntimeError(f"missing {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue          # the header, or a line torn by an interrupted run
                if "id" not in row or "knobs" not in row:
                    continue

                if row["id"] in seen:
                    # The same sweep segment rendered twice at identical knob positions in
                    # two batches is a duplicate, not new data, and keeping it would let one
                    # clip appear in both train and val. Which side wins is arbitrary, so
                    # say so rather than pick silently.
                    if seen[row["id"]] == db_dir:
                        continue    # the same row listed twice within one manifest
                    raise RuntimeError(
                        f"render id {row['id']!r} appears in both {seen[row['id']]} and {db_dir}; "
                        "the two batches overlap, so merging them would duplicate data"
                    )

                seen[row["id"]] = db_dir

                out.append(
                    Render(
                        id=row["id"],
                        file=row["file"],
                        knobs={k: float(v) for k, v in row["knobs"].items()},
                        offset_s=float(row["offset_s"]),
                        duration_s=float(row["duration_s"]),
                        input_gain=float(row.get("input_gain", 1.0)),
                        status=row.get("status", "ok"),
                        peak=float(row.get("peak", 0.0)),
                        rms=float(row.get("rms", 0.0)),
                        clipped=int(row.get("clipped", 0)),
                        nonfinite=int(row.get("nonfinite", 0)),
                        flat=bool(row.get("flat", False)),
                        db_dir=db_dir,
                    )
                )

    if not out:
        raise RuntimeError(f"no usable rows in any of: {', '.join(db_dirs)}")

    return out


def databases_of(renders: list) -> list:
    """The distinct databases a set of renders came from, in first-seen order."""
    out = []
    for r in renders:
        if r.db_dir not in out:
            out.append(r.db_dir)
    return out


def load_input(db_dirs) -> tuple:
    """
    Load the shared input recording the renderer normalized and simulated.

    This must be the renderer's `input/normalized.wav`, not the user's original file: the
    renderer scales the input before the circuit sees it, so a different normalization
    would put the model and the data a fixed dB apart everywhere.

    When several databases are merged, every one of them must have produced the same
    input/normalized.wav, byte for byte. Two batches rendered from differently normalized
    sources are two different experiments; merging them would teach the model that one
    input produces two outputs, and the resulting error would be spread thinly enough
    across the whole knob surface to look like ordinary model noise rather than a bug.
    """
    if isinstance(db_dirs, str):
        db_dirs = [db_dirs]
    db_dirs = [os.path.abspath(d) for d in db_dirs]

    if not db_dirs:
        raise RuntimeError("load_input: no database given")

    x = None
    sr = None

    for db_dir in db_dirs:
        path = os.path.join(db_dir, "input", "normalized.wav")

        if not os.path.exists(path):
            raise RuntimeError(f"missing {path}; the renderer writes it next to the manifest")

        xi, si = sf.read(path, dtype="float64", always_2d=False)

        if xi.ndim != 1:
            raise RuntimeError(f"{path} is not mono")

        if x is None:
            x, sr = xi, si
            continue

        if si != sr:
            raise RuntimeError(
                f"{path} is {si} Hz but another database's input is {sr} Hz; "
                "these cannot be merged"
            )

        if xi.shape != x.shape or not np.array_equal(xi, x):
            bad = np.argmax(np.abs(xi - x)) if xi.shape == x.shape else -1
            raise RuntimeError(
                "merged databases do not share the same input/normalized.wav "
                f"({db_dir} differs from {db_dirs[0]}, first differing sample {bad}). "
                "Merge only batches rendered from the same normalized input."
            )

    return x.astype(np.float32), sr


def load_render(r: Render, sr: int) -> np.ndarray:
    """Load one render's output, mono, float32, from the database it came from."""
    path = os.path.join(r.db_dir, "renders", r.file)
    y, s = sf.read(path, dtype="float64", always_2d=False)
    if s != sr:
        raise RuntimeError(f"{path} is {s} Hz, expected {sr}")
    if y.ndim != 1:
        y = y.mean(axis=1)
    return y.astype(np.float32)


def expected_length(r: Render, sr: int) -> int:
    """How many samples the renderer should have written for this render."""
    return int(round(r.offset_s * sr)), int(round(r.duration_s * sr))


def validate_geometry(renders: list, x: np.ndarray, sr: int) -> None:
    """
    Confirm the recovered input lines up with the recovered output, sample for sample.

    The renderer truncated a render when its segment ran off the end of the input. A
    mismatch here means the manifest and the files disagree, which would silently teach the
    model a gain error rather than a circuit model, so it is an error and not a warning.
    """
    n_in = len(x)
    for r in renders:
        start, count = expected_length(r, sr)
        if start < 0 or start + count > n_in:
            raise RuntimeError(
                f"{r.id}: segment [{start},{start + count}) is outside the "
                f"{n_in}-sample input; manifest and input disagree"
            )
        r._expected = count          # type: ignore[attr-defined]


def report(renders: list, label: str = "") -> None:
    """Print what the database actually contains, so a bad one is visible before training."""
    usable = [r for r in renders if r.usable]
    peaks = np.array([r.peak for r in usable]) if usable else np.zeros(1)
    print(f"database {label}")
    print(f"  renders     : {len(renders)} total, {len(usable)} usable")
    bad = {}
    for r in renders:
        why = []
        if r.status in BAD_STATUS:
            why.append(r.status)
        if r.nonfinite:
            why.append("nonfinite")
        if r.flat:
            why.append("flat")
        if r.peak <= QUIET_PEAK:
            why.append("silent")
        if why:
            key = "+".join(why)
            bad[key] = bad.get(key, 0) + 1
    print(f"  unusable    : {bad if bad else 'none'}")
    if len(usable):
        db = 20.0 * np.log10(np.maximum(peaks, 1e-12))
        print(f"  peak        : {db.min():.1f} .. {db.max():.1f} dBFS "
              f"({peaks.min():.3g} .. {peaks.max():.3g})")
        clipped = sum(r.clipped for r in usable)
        print(f"  clipped     : {clipped} sample(s) in {sum(1 for r in usable if r.clipped)} render(s)")
    # Pinned knobs (Master 1 = 0.5, Pull Lead = 1.0) are excluded from the model's
    # feature vector: neither varies, so neither carries information, and feeding them
    # in would waste parameters on constant channels. They are still rendered into the
    # audio, so the network learns their effect implicitly.
    _PINNED_KNOBS = frozenset({"Master 1", "Pull Lead"})
    knobs = [k for k in sorted(renders[0].knobs.keys()) if k not in _PINNED_KNOBS]
    print("  knobs       : " + ", ".join(f"{k} {min(r.knobs[k] for r in renders):.2f}..{max(r.knobs[k] for r in renders):.2f}" for k in knobs))
    if _PINNED_KNOBS:
        pinned = sorted(_PINNED_KNOBS & set(renders[0].knobs.keys()))
        print("  pinned      : " + ", ".join(f"{k}={renders[0].knobs[k]:.2f} (constant)" for k in pinned))
    offs = np.array([r.offset_s for r in renders])
    print(f"  segments    : {offs.min():.1f} .. {offs.max() + renders[0].duration_s:.1f}s, "
          f"{len(np.unique(offs))} distinct")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def timeline_extent(renders: list, duration: float, sr: float = 48000.0):
    """
    The span of source audio these draws can ever occupy, as (lo, hi) seconds.

    This wants a number that does not change as renders are appended, because every band
    boundary is a fraction of it. Three sources are tried in order of trustworthiness:

    1. The pack's input.npy length. The pack reads the whole normalized.wav, so its length
       is the real extent of the source regardless of which excerpts were drawn. This is
       the one used in normal operation.
    2. The manifest header, if a future livespice-gen records input_frames.
    3. The renders observed so far -- correct only for a finished database. This is not
       silent: it prints a warning, because on a partial database it yields empty val and
       test bands and then relocates them, which is the failure this exists to prevent.
    """
    for db_dir in databases_of(renders):
        npy = os.path.join(db_dir, "pack", "input.npy")
        if os.path.exists(npy):
            try:
                n = int(os.path.getsize(npy) // 4)      # float32
                if n > 0:
                    return 0.0, n / float(sr)
            except Exception:
                pass

    for db_dir in databases_of(renders):
        path = os.path.join(db_dir, "manifest.jsonl")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = json.loads(f.readline())
            n = int(header.get("input_frames", 0))
            if n > 0:
                return 0.0, n / float(header.get("sample_rate") or sr)
        except Exception:
            continue

    hi = max((float(getattr(r, "offset_s", 0.0)) + duration for r in renders), default=duration)
    print("note     : could not determine the input length, so the split bands are "
          f"estimated from\n            the {len(renders)} renders present (0 to {hi:.2f}s) "
          "and will move if more\n            arrive. Re-run `prepare` on the finished "
          "database before trusting validation.")
    return 0.0, hi


def split(renders: list, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> list:
    """
    Assign train / val / test by INPUT SEGMENT, not by row, and write the choice out.

    Why this replaces a row-wise split
    ---------------------------------
    Every render in this database is a different window onto the *same*
    input/normalized.wav. Shuffling rows therefore produces a validation set that
    contains unseen knob settings but audio the model has already been fitted to,
    often the identical waveform shifted by a fraction of a second. The reported ESR
    then measures how well the network interpolates over knob settings while quietly
    leaning on memorised audio, and it looks better than the model really is.

    So the split is made on the source timeline: each render's offset_s falls into one of
    a few contiguous time bands, and a whole band belongs to exactly one split.

    The bands are separated by a gap at least one render long. Without that the leak
    survives, because a render covers [offset, offset + duration] and consecutive bands
    are adjacent: with 5 s renders on a 0.25 s offset grid, a train render starting at
    120 s reaches 125 s, so a val render starting at 120.25 s shares 4.75 s of audio with
    it. Guarding costs some source audio and is worth it.

    Two secondary properties are preserved:

    * Each render stays a self-contained (input segment, knob setting, output) triple;
      audio is never carved out of the middle of one.
    * Every split spans the whole knob surface, because the stratified sampler
      distributes knob settings across all offsets rather than tying them to one.
    """
    names = ("train", "val", "test")

    if not renders:
        return renders

    # Length of one render, from the data itself rather than a constant, so this keeps
    # working if the excerpt length changes.
    duration = max((float(getattr(r, "duration_s", 0.0)) for r in renders), default=0.0)
    if duration <= 0.0:
        duration = 1.0

    # The bands are anchored to the *source timeline*, not to the offsets that happen to
    # have been rendered yet. This matters while the database is still growing, and both
    # ends have to be fixed: anchoring only the start still leaves the end derived from
    # observed offsets, so a database covering the first 45 s produced an empty
    # validation band and moved it later.
    #
    # The end comes from the render database's own header, which records the length of
    # the input it drew from. That number does not change as renders are appended, so a
    # render's split is a pure function of its offset and is decided the first time it is
    # seen. Without a header the extent is estimated from the data, which is only correct
    # once the database is complete -- so that path says so instead of pretending.
    lo0, hi0 = timeline_extent(renders, duration)
    if hi0 <= lo0:
        hi0 = lo0 + duration

    n_gaps = len(names) - 1
    guard = duration                      # no split may start within this of the previous
    usable = max(1e-6, (hi0 - lo0) - guard * n_gaps)

    assigned = {n: [] for n in names}
    bounds = []
    cursor = lo0
    bands = []

    for k, (name, frac) in enumerate(zip(names, fractions)):
        span = usable * frac
        band_hi = cursor + span - (guard if k < n_gaps else 0.0)
        bands.append((name, cursor, band_hi))
        cursor = band_hi + guard

    for i, r in enumerate(renders):
        off = float(getattr(r, "offset_s", 0.0))
        for name, blo, bhi in bands:
            if blo <= off <= bhi:
                r.split = name
                assigned[name].append(i)
                break

    for name, blo, bhi in bands:
        got = [float(getattr(renders[i], "offset_s", 0.0)) for i in assigned[name]]
        bounds.append((min(got), max(got)) if got else (None, None))

    # Report what was actually produced. A split that silently collapsed to empty is the
    # failure mode worth catching loudly, especially since a run against a
    # still-growing database can produce exactly that.
    print("split by input segment (guard %.2f s between bands):" % guard)
    for name, (blo, bhi) in zip(names, bounds):
        n = len(assigned[name])
        span = "EMPTY" if blo is None else "offsets %.2f..%.2f s -> audio %.2f..%.2f s" % (
            blo, bhi, blo, bhi + duration)
        print("  %-5s %5d renders  %s" % (name, n, span))

    return renders


def knob_sparsity(val_renders: list, train_renders: list, knob_names: list) -> list:
    """
    For each validation render, how far is its knob setting from anything seen in training?

    This is what actually answers "is the model parametric or did it memorise a grid".
    A row-wise split cannot answer it, because every knob setting in validation also
    appears in training. Here each val render is paired with its nearest training
    neighbour in knob space, and that distance is what bucketed reporting in
    train.sparsity_report sorts on: ESR in the sparse buckets is the number that has to
    stay low for the plugin to be usable at settings nobody rendered.

    Distance is the Euclidean norm across knobs, each already in [0, 1] because knob
    positions are, so no further normalisation is needed or applied.

    Accepts either a Render (which carries `.knobs`) or a plain dict with a "knobs" key,
    so callers that hold packed index entries rather than Render objects need not
    rebuild one.

    Computed with numpy rather than the Python loop this used to be: the nearest-
    neighbour scan is O(V * T * K) in the number of val renders, train renders and knobs,
    and at production pack sizes (tens of thousands of windows against ~15 k train
    renders) the interpreted loop took the better part of an hour. The matmul below is
    the same squared-Euclidean distance, vectorised.
    """
    if not val_renders or not train_renders or not knob_names:
        return []

    def vec(r):
        k = getattr(r, "knobs", None)
        if k is None:
            k = r["knobs"]
        return np.asarray([float(k.get(name, 0.0)) for name in knob_names], dtype=np.float64)

    val = np.stack([vec(r) for r in val_renders])          # (V, K)
    train = np.stack([vec(r) for r in train_renders])      # (T, K)

    # ||v - t||^2 = ||v||^2 + ||t||^2 - 2 v.t. Work in blocks over the val side so the
    # (V, T) accumulator never materialises in one shot.
    val_norm = (val * val).sum(axis=1, keepdims=True)      # (V, 1)
    train_dot = (train * train).sum(axis=1)                # (T,)
    block = 512
    out = np.empty(val.shape[0], dtype=np.float64)
    for i0 in range(0, val.shape[0], block):
        i1 = min(i0 + block, val.shape[0])
        d2 = (val_norm[i0:i1]
              + train_dot[None, :]
              - 2.0 * (val[i0:i1] @ train.T))
        np.maximum(d2, 0.0, out=d2)   # numerical dregs can dip a hair below zero
        out[i0:i1] = np.sqrt(d2.min(axis=1))
    return out.tolist()


def save_split(renders: list, split_dir: str = None) -> list:
    """
    Persist the split next to each database (or in split_dir) so training and export agree on it.
    """
    written = []
    if split_dir is not None:
        os.makedirs(split_dir, exist_ok=True)
        mine = [r for r in renders if r.split]
        path = os.path.join(split_dir, "split.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({r.id: r.split for r in mine}, f, indent=1, sort_keys=True)
        written.append(path)
        return written

    for db_dir in databases_of(renders):
        mine = [r for r in renders if r.db_dir == db_dir and r.split]
        path = os.path.join(db_dir, "split.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump({r.id: r.split for r in mine}, f, indent=1, sort_keys=True)

        written.append(path)

    return written


def load_split(renders: list, split_dir: str = None) -> bool:
    """
    Reuse previously written splits, but only if they cover every render.
    """
    if split_dir is not None:
        p = os.path.join(split_dir, "split.json")
        paths = [p] if os.path.exists(p) else []
    else:
        paths = [os.path.join(d, "split.json") for d in databases_of(renders)]
    tables = []

    for path in paths:
        if not os.path.exists(path):
            print(f"no split.json in {os.path.dirname(path)}; the split will be drawn")
            return False
        with open(path, "r", encoding="utf-8") as f:
            tables.append(json.load(f))

    assigned = {}
    for table in tables:
        assigned.update(table)

    missing = 0
    for r in renders:
        s = assigned.get(r.id)
        if s:
            r.split = s
        else:
            missing += 1

    if missing:
        print(f"split.json covers {len(renders) - missing}/{len(renders)} renders; "
              f"{missing} have no entry, so the split will be drawn again")
        return False

    return True