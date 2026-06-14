"""
src/eval/seq_baselines.py

The "why not just a sequence model?" control, made FAIR (no strawman).

A reviewer will object that a Transformer with *learned absolute* positional embeddings fails to
extrapolate only because positions past the trained length are untrained. So here we give the
Transformer its best shot at length generalization and path integration:

  - xf-learned (mean-pool)   : the naive default (learned absolute positions).
  - xf-sinusoid (mean-pool)  : sinusoidal positions — deterministic and defined at EVERY length.
  - xf-nope-sum (sum-pool)   : NO positional encoding (permutation-invariant — correct for a
                               commutative path sum) + SUM pooling — practically built to integrate.
  - GRU                      : recurrent integrator.
  - grid code (ours)         : the fixed velocity-driven grid code.

Same faithful 2-D random-walk task, same readout/budget, train on mixed {6,8,10,12}, test to 4x
(src/eval/extrapolation.py). Distance exact-acc + position error vs length, mean ± 95% CI.

The honest question: does ANY standard sequence model extrapolate trajectory MAGNITUDE to unseen
lengths the way the grid code does — and at what positional/pooling cost? Writes
results/seq_baselines.json + results/seq_baselines.svg.

    python -m src.eval.seq_baselines --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.eval.extrapolation import make_batch, head, metrics, ci95, GRURep
from src.eval.ablations import GridRepN, train_eval


def sinusoid(T, d):
    pos = torch.arange(T).unsqueeze(1).float()
    i = torch.arange(0, d, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SeqTransformer(nn.Module):
    def __init__(self, pos="sinusoid", pool="mean", d=64, layers=2, heads=4, maxlen=80):
        super().__init__()
        self.pos, self.pool, self.d = pos, pool, d
        self.inp = nn.Linear(2, d)
        if pos == "learned":
            self.posemb = nn.Parameter(torch.randn(maxlen, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = head(d)

    def forward(self, v, disp):
        T = v.shape[1]
        x = self.inp(v)
        if self.pos == "learned":
            x = x + self.posemb[:T].unsqueeze(0)
        elif self.pos == "sinusoid":
            x = x + sinusoid(T, self.d).to(x.device).unsqueeze(0)
        # "none" -> no positional information (permutation-invariant)
        h = self.tr(x)
        pooled = h.sum(1) if self.pool == "sum" else h.mean(1)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test_lengths", type=int, nargs="+", default=[8, 16, 24, 48])
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    MIX = [6, 8, 10, 12]
    TL = a.test_lengths

    variants = {
        "grid code (ours)": (lambda: GridRepN(n_modules=6)),
        "xf-learned (mean)": (lambda: SeqTransformer(pos="learned", pool="mean")),
        "xf-sinusoid (mean)": (lambda: SeqTransformer(pos="sinusoid", pool="mean")),
        "xf-NoPE (sum)": (lambda: SeqTransformer(pos="none", pool="sum")),
        "GRU": GRURep,
    }
    print(f"FAIR SEQUENCE-MODEL BASELINES (n={a.seeds} seeds; mean ± 95% CI)\n"
          f"train mixed {MIX}, test {TL}\n" + "=" * 68, flush=True)

    agg = {}
    for label, mk in variants.items():
        per_seed = [train_eval(mk, s, MIX, TL) for s in seeds]
        agg[label] = {}
        for T in TL:
            agg[label][T] = {}
            for met in ("distance_exact_acc", "pos_decode_error", "bearing_acc"):
                m, ci = ci95([p[T][met] for p in per_seed])
                agg[label][T][met] = {"mean": m, "ci95": ci}
        d = agg[label]
        print(f"  {label:20} distance " +
              "  ".join(f"T{T}:{d[T]['distance_exact_acc']['mean']:.0%}±{d[T]['distance_exact_acc']['ci95']:.0%}"
                        for T in TL), flush=True)

    out = {"n_seeds": a.seeds, "train_lengths": MIX, "test_lengths": TL, "results": agg}
    os.makedirs("results", exist_ok=True)
    with open("results/seq_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_seq(agg, TL, "results/seq_baselines.svg")
    print("\nwrote results/seq_baselines.json and results/seq_baselines.svg", flush=True)


PALETTE = ["#e6550d", "#3b528b", "#21908c", "#756bb1", "#9aa5b8"]


def svg_seq(agg, Ts, out):
    pad = 56; pw = 320; ph = 230
    W = pad + pw + 200; H = pad + ph + 70
    labels = list(agg.keys())
    def X(T): return pad + Ts.index(T) / (len(Ts) - 1) * pw
    def Y(v): return (pad + 20) + ph - v * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Fair sequence-model baselines: only the grid code holds the magnitude</text>')
    e.append('<text x="28" y="43" font-size="10.5" fill="#5b6b8c">distance exact-acc vs path length; '
             'train &#8804;12, test to 4&#215; &#183; mean &#177; 95% CI</text>')
    oy = pad + 20
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.0, 0.25, 0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad}" y1="{Y(vv):.1f}" x2="{pad+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-7}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for T in Ts:
        e.append(f'<text x="{X(T):.1f}" y="{oy+ph+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">T={T}</text>')
    for i, lab in enumerate(labels):
        col = PALETTE[i % len(PALETTE)]
        m = agg[lab]
        band_t = " ".join(f"{X(T):.1f},{Y(m[T]['distance_exact_acc']['mean']+m[T]['distance_exact_acc']['ci95']):.1f}" for T in Ts)
        band_b = " ".join(f"{X(T):.1f},{Y(m[T]['distance_exact_acc']['mean']-m[T]['distance_exact_acc']['ci95']):.1f}" for T in reversed(Ts))
        e.append(f'<polygon points="{band_t} {band_b}" fill="{col}" opacity="0.12"/>')
        pts = " ".join(f"{X(T):.1f},{Y(m[T]['distance_exact_acc']['mean']):.1f}" for T in Ts)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        for T in Ts:
            e.append(f'<circle cx="{X(T):.1f}" cy="{Y(m[T]["distance_exact_acc"]["mean"]):.1f}" r="3" fill="{col}"/>')
    ly = oy + 6
    for i, lab in enumerate(labels):
        e.append(f'<rect x="{pad+pw+12}" y="{ly}" width="14" height="5" fill="{PALETTE[i % len(PALETTE)]}"/>')
        e.append(f'<text x="{pad+pw+30}" y="{ly+6}" font-size="10.5" fill="#28324a">{lab}</text>'); ly += 19
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
