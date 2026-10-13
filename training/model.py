# File: model.py
"""
A knob-conditioned recurrent amp model, and its RTNeural export.

Shape of the problem: the plugin gets three continuous knobs and must answer one output
sample at a time, forever, inside an audio callback. That rules out anything needing a
lookahead window, a windowed FFT, or a custom C++ layer, so the model is a plain
sample-by-sample recurrent stack that RTNeural already implements.

Conditioning is by input concatenation rather than FiLM or a hypernetwork. The knob values
are folded into the model's input vector, which means the plugin builds a 3-float array per
sample and hands it to `ModelT<float, 3, 1, ...>`. FiLM would have been slightly more
expressive and would have required hand-written C++ that no existing tooling produces.

What the input channels are, and why:

    x        The audio, unmodified. Volume is a passive output attenuator on this circuit
             and the clipping stage sits *before* it, so the nonlinearity never sees an
             attenuated signal and neither should the model.
    v        Volume, as its own channel. It must be separate: `x * v` is ambiguous, because
             0.1 means "loud input, volume nearly down" in one case and "quiet input, volume
             full" in the other, and those need different outputs. Folding the gain into the
             audio hides that and costs accuracy exactly where the knob is being used.
    s        Sustain, which drives a clipping stage. The gain it applies is entangled with how
             hard it clips, so it conditions the state rather than the sample.
    t        Tone, a filter. A filter is not a function of the present sample alone, so it has
             to condition the state too.

The `x, v, s, t` layout follows ChowCentaur's conditioned models (`centaur.json`,
`TimeDistributed(Dense) -> GRU -> Dense`, inputs `[audio, gain, dt]`), the one real
precedent for conditioning a pedal this way. Its shipped models do not use conditioning --
they are five single-input models crossfaded between -- so this is the harder path, and the
reason to keep going is that DAFx-19 conditioned a Big Muff of this very type successfully.

Two constraints are load-bearing and easy to break by accident:

  * Gate order. PyTorch lays a GRU's 3H rows out as (r, z, n). RTNeural lays them out as
    (z, r, n). `export_rtneural` permutes them. If that permutation is dropped the model
    loads without complaint and sounds wrong, which is why `verify_export` exists.
  * Bias arity. RTNeural reads a GRU bias as a 2 x 3H array, with the first row added to
    the input projection and the second to the recurrent one. PyTorch splits the same bias
    across `bias_ih_l0` and `bias_hh_l0`, which is exactly the same split.

One more, learned the hard way from ChowCentaur: an activation has to be written into the
JSON. Its conditioned models were trained in Keras with `tanh` on the first dense, but the
export omitted the `activation` key, so RTNeural ran the layer linear. Training and
inference disagreed and nobody noticed, because nothing errors.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn


class KnobGRU(nn.Module):
    """
    dense -> tanh -> GRU -> dense, conditioned by concatenating knob values into the input.

    The conditioning channels are constant across time, so they are passed once per batch
    and broadcast here rather than materialised by the caller.
    """

    def __init__(self, n_features: int = 4, hidden: int = 32, n_layers: int = 1):
        super().__init__()
        if n_layers != 1:
            # RTNeural's JSON loader walks a flat layer list and does not carry a per-layer
            # index in the recurrent weights, so stacking would produce weights it reads
            # back wrong. One layer it is; depth is not what this model is short of.
            raise NotImplementedError("n_layers > 1 has no RTNeural export")
        self.n_features = n_features
        self.hidden = hidden
        self.dense_in = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.dense_out = nn.Linear(hidden, 1)

    @staticmethod
    def build_features(x, cond):
        """
        Compose the per-sample input from the audio and the per-render knob values.

        `x` is (B, T) and `cond` is (B, F-1) holding the knobs in FEATURE_ORDER minus the
        audio: volume, then any other conditioning knobs. They are constant across time, so
        they are broadcast here rather than materialised by the caller.
        """
        audio = x.unsqueeze(-1)                                # (B,T,1)
        knobs = cond.unsqueeze(1).expand(-1, x.shape[1], -1)   # (B,T,F-1)
        return torch.cat([audio, knobs], dim=-1)               # (B,T,F)

    def forward(self, x, cond, state=None):
        """
        x (B,T), cond (B,F-1), state (1,B,H) or None -> (B,T,1), new state (1,B,H).

        PyTorch's GRU returns its hidden state as a (num_layers * num_directions, B, H)
        tensor rather than a per-layer tuple, so the layer axis has to be kept: truncated
        BPTT feeds this straight back in on the next chunk.
        """
        feat = self.build_features(x, cond)
        h = torch.tanh(self.dense_in(feat))
        out, state = self.gru(h, state)
        return self.dense_out(out), state

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# RTNeural export
# ---------------------------------------------------------------------------

def _permute_gates(w, hidden):
    """
    Reorder a (3H, N) PyTorch GRU weight matrix from (r, z, n) to RTNeural's (z, r, n).

    Only the block axis moves; within a block the rows are untouched.
    """
    return torch.cat([w[hidden:2 * hidden], w[0:hidden], w[2 * hidden:3 * hidden]], dim=0)


def export_rtneural(model: KnobGRU, path: str, metadata=None):
    """
    Write the model as RTNeural JSON and return the path.

    Layout, read back through `model_loader.h`:
      in_shape             [None, None, n_features]; RTNeural only reads the last entry.
      dense                weights[0] is [in][out], weights[1] is a flat [out] bias.
      gru                  weights[0] is [in][3*out], weights[1] is [out][3*out],
                           weights[2] is 2 x [3*out] (input bias, then recurrent bias).
    Both dense matrices are stored transposed relative to PyTorch, because RTNeural indexes
    them output-major while PyTorch indexes them input-major.
    """
    was_training = model.training
    model.eval()

    H = model.hidden
    F = model.n_features

    layers = [
        {
            "type": "dense",
            "shape": [None, None, H],
            "activation": "tanh",
            "weights": [
                model.dense_in.weight.detach().cpu().numpy().T.tolist(),   # (F,H)
                model.dense_in.bias.detach().cpu().numpy().tolist(),       # (H,)
            ],
        },
        {
            "type": "gru",
            "shape": [None, None, H],
            "weights": [
                # weight_ih_l0 is (3H, H); RTNeural wants (H, 3H).
                _permute_gates(model.gru.weight_ih_l0.detach(), H).cpu().numpy().T.tolist(),
                # weight_hh_l0 is (3H, H); RTNeural wants (H, 3H).
                _permute_gates(model.gru.weight_hh_l0.detach(), H).cpu().numpy().T.tolist(),
                [
                    _permute_gates(model.gru.bias_ih_l0.detach(), H).cpu().numpy().tolist(),
                    _permute_gates(model.gru.bias_hh_l0.detach(), H).cpu().numpy().tolist(),
                ],
            ],
        },
        {
            "type": "dense",
            "shape": [None, None, 1],
            "weights": [
                model.dense_out.weight.detach().cpu().numpy().T.tolist(),  # (H,1)
                model.dense_out.bias.detach().cpu().numpy().tolist(),      # (1,)
            ],
        },
    ]

    doc = {"in_shape": [None, None, F], "layers": layers}
    if metadata:
        # Not read by RTNeural. Kept in the file so a shipped model says what it is and how
        # it was made, which is the difference between a model and an artifact.
        doc["metadata"] = metadata

    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    if was_training:
        model.train()
    return path


def verify_export(model: KnobGRU, path: str, x: torch.Tensor, cond: torch.Tensor, tol=2e-4):
    """
    Replay the exported weights in numpy, using RTNeural's own equations, and compare.

    This is not a formality. The gate permutation and the dense transpose are both silent
    failures: the JSON parses, the shapes are all legal, and the result is a network that
    was never trained. Re-implementing the layer maths here in numpy, straight from the
    comments in `gru_eigen.h`, is the only way to catch that without a C++ harness in the
    loop.

    The reference is computed in float64 on purpose. Comparing the float64 replay against a
    float32 CUDA forward pass measures the wrong thing: PyTorch runs fp32 matmuls through
    TF32 on Ampere and later, which keeps about 10 mantissa bits and lands around 3e-4
    relative error. That is the same order as `tol`, so a perfectly correct export fails the
    check while the real bugs -- an unpermuted gate order costs ~1.7e-1, a dropped tanh
    ~1.7e-2 -- sit one to three orders of magnitude above the tolerance and are still
    caught with room to spare.

    Returns (max_abs_error, ok).
    """
    import copy
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    L = {l["type"] + str(i): l for i, l in enumerate(doc["layers"])}
    d1 = doc["layers"][0]
    gru = doc["layers"][1]
    d2 = doc["layers"][2]

    W1 = np.array(d1["weights"][0], dtype=np.float64)     # (F,H)
    b1 = np.array(d1["weights"][1], dtype=np.float64)     # (H,)
    Wv = np.array(gru["weights"][0], dtype=np.float64)    # (H,3H)
    Uv = np.array(gru["weights"][1], dtype=np.float64)    # (H,3H)
    Bv = np.array(gru["weights"][2], dtype=np.float64)    # (2,3H)
    W2 = np.array(d2["weights"][0], dtype=np.float64)     # (H,1)
    b2 = np.array(d2["weights"][1], dtype=np.float64)     # (1,)

    H = W1.shape[1]
    xn = x.detach().cpu().numpy().astype(np.float64)
    cn = cond.detach().cpu().numpy().astype(np.float64)
    B, T = xn.shape

    # Read the activation out of the JSON rather than assuming it. RTNeural applies whatever
    # the file says, so a missing key is a layer running linear -- and that is precisely the
    # kind of export bug this function exists to catch.
    act = d1.get("activation", "")

    def apply_act(v):
        if act == "tanh":
            return np.tanh(v)
        if act == "sigmoid":
            return 1.0 / (1.0 + np.exp(-v))
        if act == "":
            return v
        raise AssertionError(f"verifier does not implement activation {act!r}")

    n_cond = cn.shape[1]
    feat = np.concatenate(
        [
            xn[:, :, None],
            np.broadcast_to(cn[:, None, :], (B, T, n_cond)),   # (B,1,C) -> (B,T,C)
        ],
        axis=-1,
    )                                    # (B,T,F)

    pred = np.zeros((B, T, 1), dtype=np.float64)
    for b in range(B):
        h = np.zeros(H, dtype=np.float64)
        for t in range(T):
            e = apply_act(feat[b, t] @ W1 + b1)            # (H,)
            alpha = Wv.T @ e + Bv[0]                       # (3H,) = W x + b_ih
            beta = Uv.T @ h + Bv[1]                        # (3H,) = U h + b_hh
            g = 1.0 / (1.0 + np.exp(-(alpha[:2 * H] + beta[:2 * H])))
            z, r = g[:H], g[H:]
            n = np.tanh(alpha[2 * H:] + r * beta[2 * H:])
            h = n + z * (h - n)                             # (1-z)*n + z*h
            pred[b, t, 0] = float(h @ W2[:, 0] + b2[0])

    with torch.no_grad():
        # deepcopy before .double(): nn.Module.double() is in-place and would mutate the
        # caller's trained model.
        m64 = copy.deepcopy(model).double().cpu()
        ref, _ = m64(x.detach().cpu().double(), cond.detach().cpu().double())
    ref = ref.cpu().numpy().astype(np.float64)

    err = float(np.abs(pred - ref).max())
    scale = float(np.abs(ref).max())
    ok = err <= tol * max(scale, 1.0)
    return err, ok