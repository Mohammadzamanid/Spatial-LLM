"""
src/eval/relational.py

ABSTRACT / RELATIONAL COGNITION — the grid map as a relational engine, not just a spatial one.

The hippocampal-entorhinal system maps not only physical space but RELATIONAL STRUCTURE — concept
spaces, ordered sets, task graphs — with the same grid/place code (Tolman-Eichenbaum Machine,
Whittington 2020; grid codes in concept space, Constantinescu 2016; relational memory, Eichenbaum).
We place an abstract ORDERED structure (items ranked 0..N-1) along a concept axis, let the SAME
velocity-driven grid cortex map it, and read relations off the metric.

Tests (all on a structure the agent maps but is taught only LOCAL comparisons):
  1. TRANSITIVE INFERENCE — train only ADJACENT comparisons (i vs i+1); infer NON-adjacent pairs
     (A>D, never seen). The metric makes B>... transitive.
  2. SYMBOLIC DISTANCE EFFECT — with neural noise, far-apart pairs are EASIER (higher accuracy) —
     the behavioural signature of an analog/spatial representation of an abstract dimension.
  3. SCHEMA TRANSFER — a NEW item set with the same ordinal structure, in a new region of the
     concept space: the learned comparison transfers zero-shot (structure abstracted from content).

Writes results/relational.json and results/relational.svg (the symbolic-distance-effect curve).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules


def grid_code(cx, pos):
    phi = cx.gains.view(-1, 1, 1) * pos.unsqueeze(0)
    return cx._grid_code(phi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)        # items in the ordered structure
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.8)   # neural noise; reveals the distance effect
    a = ap.parse_args()
    torch.manual_seed(0)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    N, D = a.N, a.spacing
    ranks = torch.arange(N).float()
    # items laid along an ABSTRACT rank axis (a concept dimension), centred; mapped by the grid cortex
    item_pos = torch.stack([ranks * D - (N - 1) * D / 2, torch.zeros(N)], -1)
    codes = grid_code(cx, item_pos)                                      # each item's grid representation

    # comparison readout: C([code_i, code_j]) -> logit P(rank_i > rank_j)
    C = nn.Sequential(nn.Linear(2 * cx.K * cx.M, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(C.parameters(), lr=1e-3)

    def feat(pairs, cd):
        a_, b_ = pairs[:, 0], pairs[:, 1]
        return torch.cat([cd[a_], cd[b_]], -1), (ranks[a_] > ranks[b_]).float()

    adj = torch.tensor([(i, i + 1) for i in range(N - 1)] + [(i + 1, i) for i in range(N - 1)])
    for _ in range(3000):                                                # train ONLY on adjacent pairs
        x, y = feat(adj, codes)
        opt.zero_grad(); F.binary_cross_entropy_with_logits(C(x).squeeze(-1), y).backward(); opt.step()

    nonadj = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) >= 2])
    allp = torch.tensor([(i, j) for i in range(N) for j in range(N) if i != j])

    @torch.no_grad()
    def acc(pairs, cd, noise=0.0, trials=8):
        a_, b_ = pairs[:, 0], pairs[:, 1]; y = (ranks[a_] > ranks[b_]).float()
        cor = 0.0
        for _ in range(trials):
            ci = cd[a_] + noise * torch.randn_like(cd[a_]); cj = cd[b_] + noise * torch.randn_like(cd[b_])
            pred = (C(torch.cat([ci, cj], -1)).squeeze(-1) > 0).float()
            cor += (pred == y).float().mean().item()
        return cor / trials

    ti_acc = acc(nonadj, codes, a.noise)                                 # transitive inference (never trained)
    adj_acc = acc(adj, codes, a.noise)
    # symbolic distance effect: accuracy by rank-distance (with neural noise)
    sde = {}
    for d in range(1, N):
        pd = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) == d])
        sde[d] = round(acc(pd, codes, a.noise, trials=16), 3)

    # schema transfer: a NEW ordered set in a different region of the concept space (same structure)
    item_pos2 = torch.stack([ranks * D - (N - 1) * D / 2 + 0.3, torch.full((N,), 1.8)], -1)
    codes2 = grid_code(cx, item_pos2)
    transfer_acc = acc(nonadj, codes2, a.noise)                          # zero-shot in the new "environment"

    out = {"N": N, "transitive_inference_acc": round(ti_acc, 3), "adjacent_trained_acc": round(adj_acc, 3),
           "schema_transfer_acc": round(transfer_acc, 3),
           "symbolic_distance_effect_acc_by_dist": sde}
    print("ABSTRACT / RELATIONAL COGNITION — grid map as a relational engine:", flush=True)
    print(f"  TRANSITIVE INFERENCE: {100*ti_acc:.0f}% on non-adjacent pairs NEVER trained "
          f"(adjacent trained pairs: {100*adj_acc:.0f}%)", flush=True)
    print(f"  SYMBOLIC DISTANCE EFFECT (acc by rank-distance, far = easier): "
          + "  ".join(f"d{d}:{sde[d]:.0%}" for d in sorted(sde)), flush=True)
    print(f"  SCHEMA TRANSFER: {100*transfer_acc:.0f}% on a NEW ordered set in a new region "
          f"(structure abstracted from content)", flush=True)

    svg_sde(sde, N, "results/relational.svg")
    os.makedirs("results", exist_ok=True)
    with open("results/relational.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/relational.json and results/relational.svg", flush=True)


def svg_sde(sde, N, out):
    W, H, pad = 460, 300, 56
    ds = sorted(sde); ys = [sde[d] for d in ds]
    def X(d): return pad + (d - 1) / (max(ds) - 1) * (W - 2 * pad)
    def Y(v): return H - pad - (v - 0.4) / 0.62 * (H - 2 * pad)        # y-range ~0.4..1.0
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Symbolic distance effect (relational inference)</text>')
    e.append('<text x="28" y="44" font-size="11.5" fill="#5b6b8c">accuracy vs rank-distance — '
             'far-apart pairs are EASIER (analog/spatial code)</text>')
    e.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#33415c"/>')
    for v in (0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad-3}" y1="{Y(v):.1f}" x2="{W-pad}" y2="{Y(v):.1f}" stroke="#e8edf5"/>')
        e.append(f'<text x="{pad-30}" y="{Y(v)+4:.1f}" font-size="10" fill="#5b6b8c">{int(v*100)}%</text>')
    e.append(f'<text x="{W/2}" y="{H-16}" font-size="11" fill="#28324a" text-anchor="middle">rank distance |i−j|  (d=1 trained; d≥2 inferred)</text>')
    pts = " ".join(f"{X(d):.1f},{Y(sde[d]):.1f}" for d in ds)
    e.append(f'<polyline points="{pts}" fill="none" stroke="#3b528b" stroke-width="2.5"/>')
    for d in ds:
        col = "#9aa5b8" if d == 1 else "#e6550d"
        e.append(f'<circle cx="{X(d):.1f}" cy="{Y(sde[d]):.1f}" r="3.5" fill="{col}"/>')
        e.append(f'<text x="{X(d):.1f}" y="{H-pad+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{d}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
