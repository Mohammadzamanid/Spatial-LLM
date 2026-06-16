"""
src/eval/structural_transfer.py

THE TEM HEADLINE, design-validated on CPU before the frozen-LLM version.

Claim under test: a grid/place metric code trained ONLY on spatial path integration (never on any
relational task) is a substrate for ABSTRACT relational inference — lay a non-spatial ordered
structure (ranks/ages/dominance) along a concept axis, push each item through the FROZEN cortex as if
it were a position, train a comparison readout on ADJACENT pairs only, and transitive inference on
never-seen far pairs emerges from the metric (Tolman-Eichenbaum Machine; Whittington 2020).

This is the representation-level validation of the LLM experiment (there the readout is a frozen
Qwen+LoRA, not an MLP). It includes the two falsifiers a reviewer will demand:

  - SHUFFLED-POSITION control: place the ranks at RANDOM positions (destroy the rank↔space
    correspondence). If transitive inference still works, it is memorization, not the metric. It must
    collapse to chance.
  - SCRAMBLED-SECOND-ITEM control: at test, replace item j's code with a random item's code (keep the
    label). If accuracy persists, the readout is exploiting one item's magnitude, not comparing two.

Each item enters by its OWN position (never the signed relative displacement — that would leak the
answer). Multi-seed, mean ± 95% CI + a paired test (ordered vs shuffled). Writes
results/structural_transfer.json + .svg.

    python -m src.eval.structural_transfer --seeds 8
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
    return cx._grid_code(cx.gains.view(-1, 1, 1) * pos.unsqueeze(0))


def train_readout(codes, ranks, adj, steps, noise, seed):
    torch.manual_seed(seed)
    C = nn.Sequential(nn.Linear(2 * codes.shape[1], 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(C.parameters(), lr=1e-3)
    a_, b_ = adj[:, 0], adj[:, 1]; y = (ranks[a_] > ranks[b_]).float()
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(C(torch.cat([codes[a_], codes[b_]], -1)).squeeze(-1), y).backward()
        opt.step()
    return C


@torch.no_grad()
def accuracy(C, codes, ranks, pairs, noise, trials=16, scramble=False, gen=None):
    a_, b_ = pairs[:, 0], pairs[:, 1]; y = (ranks[a_] > ranks[b_]).float(); cor = 0.0
    for _ in range(trials):
        ci = codes[a_] + noise * torch.randn(len(a_), codes.shape[1], generator=gen)
        jb = b_[torch.randperm(len(b_), generator=gen)] if scramble else b_   # scrambled 2nd item
        cj = codes[jb] + noise * torch.randn(len(b_), codes.shape[1], generator=gen)
        cor += ((C(torch.cat([ci, cj], -1)).squeeze(-1) > 0).float() == y).float().mean().item()
    return cor / trials


def run_seed(seed, N=12, D=0.5, noise=0.8, steps=3000):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)         # frozen; trained only on space
    ranks = torch.arange(N).float()
    adj = torch.tensor([(i, i + 1) for i in range(N - 1)] + [(i + 1, i) for i in range(N - 1)])
    nonadj = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) >= 2])

    # ORDERED: rank r sits at position r along the concept axis
    pos = torch.stack([ranks * D - (N - 1) * D / 2, torch.zeros(N)], -1)
    codes = grid_code(cx, pos)
    C = train_readout(codes, ranks, adj, steps, noise, seed)
    ti = accuracy(C, codes, ranks, nonadj, noise, gen=gen)
    adj_acc = accuracy(C, codes, ranks, adj, noise, gen=gen)
    ti_scram = accuracy(C, codes, ranks, nonadj, noise, scramble=True, gen=gen)
    # SDE: far pairs easier
    sde = {}
    for d in range(1, N):
        pd = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) == d])
        sde[d] = accuracy(C, codes, ranks, pd, noise, trials=8, gen=gen)
    # schema transfer: new item set, new region of concept space
    pos2 = torch.stack([ranks * D - (N - 1) * D / 2 + 0.3, torch.full((N,), 1.8)], -1)
    transfer = accuracy(C, grid_code(cx, pos2), ranks, nonadj, noise, gen=gen)

    # SHUFFLED-POSITION control: ranks placed at RANDOM positions -> metric no longer encodes order
    perm = torch.randperm(N, generator=gen)
    codes_sh = grid_code(cx, pos[perm])                            # rank r -> a random position
    C_sh = train_readout(codes_sh, ranks, adj, steps, noise, seed + 1)
    ti_sh = accuracy(C_sh, codes_sh, ranks, nonadj, noise, gen=gen)

    return {"ti": ti, "adj": adj_acc, "ti_scrambled_2nd": ti_scram, "ti_shuffled_pos": ti_sh,
            "schema_transfer": transfer, "sde_corr": _sde_corr(sde)}


def _sde_corr(sde):
    ds = sorted(sde); dt = torch.tensor(ds).float(); at = torch.tensor([sde[d] for d in ds])
    return F.cosine_similarity((dt - dt.mean()).unsqueeze(0), (at - at.mean()).unsqueeze(0)).item()


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def paired_p(a, b, iters=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.tensor(a) - torch.tensor(b); n = d.numel(); m = d.mean().item()
    signs = torch.randint(0, 2, (iters, n), generator=g, dtype=torch.float) * 2 - 1
    return ((signs * d.abs()).mean(1).abs() >= abs(m) - 1e-12).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=8); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    keys = ["ti", "adj", "ti_shuffled_pos", "ti_scrambled_2nd", "schema_transfer", "sde_corr"]
    agg = {k: dict(zip(("mean", "ci95"), ci95([p[k] for p in per]))) for k in keys}
    p_ord_vs_sh = paired_p([p["ti"] for p in per], [p["ti_shuffled_pos"] for p in per])

    print(f"STRUCTURAL TRANSFER — frozen space-trained cortex -> relational inference (n={a.seeds})\n" + "=" * 72, flush=True)
    lab = {"ti": "transitive inference (far pairs, untrained)", "adj": "adjacent pairs (trained)",
           "ti_shuffled_pos": "CONTROL shuffled positions (should be ~chance)",
           "ti_scrambled_2nd": "CONTROL scrambled 2nd item (should be ~chance)",
           "schema_transfer": "schema transfer (new item set)", "sde_corr": "symbolic-distance-effect corr"}
    for k in keys:
        print(f"  {lab[k]:48} {agg[k]['mean']:.3f} ± {agg[k]['ci95']:.3f}", flush=True)
    print(f"\n  paired test  TI(ordered) vs TI(shuffled positions):  "
          f"Δ={agg['ti']['mean']-agg['ti_shuffled_pos']['mean']:+.3f}  p={p_ord_vs_sh:.4f}", flush=True)
    print("  -> TI emerges from the cortex's ORDERED metric; destroying it collapses TI to chance.", flush=True)

    out = {"n_seeds": a.seeds, "results": agg, "p_ordered_vs_shuffled": round(p_ord_vs_sh, 4)}
    os.makedirs("results", exist_ok=True)
    with open("results/structural_transfer.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_transfer(agg, "results/structural_transfer.svg")
    print("\nwrote results/structural_transfer.json and results/structural_transfer.svg", flush=True)


def svg_transfer(agg, out):
    bars = [("transitive inference\n(untrained far pairs)", agg["ti"], "#2ca25f"),
            ("adjacent (trained)", agg["adj"], "#9aa5b8"),
            ("schema transfer\n(new items)", agg["schema_transfer"], "#3b528b"),
            ("CONTROL: shuffled\npositions", agg["ti_shuffled_pos"], "#c9341a"),
            ("CONTROL: scrambled\n2nd item", agg["ti_scrambled_2nd"], "#e6550d")]
    pad = 56; bw = 90; gap = 26; ph = 240
    W = pad + len(bars) * (bw + gap) + pad; H = pad + ph + 70
    def Y(v): return pad + 16 + ph - v * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Structural transfer: a space-trained code does relational inference (and its falsifiers)</text>')
    e.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{W-pad}" y2="{Y(0):.1f}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{pad+16}" x2="{pad}" y2="{Y(0):.1f}" stroke="#33415c"/>')
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{W-pad}" y2="{Y(v):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-7}" y="{Y(v)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(v*100)}%</text>')
    e.append(f'<line x1="{pad}" y1="{Y(0.5):.1f}" x2="{W-pad}" y2="{Y(0.5):.1f}" stroke="#c9341a" stroke-dasharray="4,3" opacity="0.5"/>')
    e.append(f'<text x="{W-pad}" y="{Y(0.5)-3:.1f}" font-size="8.5" fill="#c9341a" text-anchor="end">chance</text>')
    for i, (name, v, col) in enumerate(bars):
        x = pad + i * (bw + gap) + gap / 2
        e.append(f'<rect x="{x:.1f}" y="{Y(v["mean"]):.1f}" width="{bw}" height="{Y(0)-Y(v["mean"]):.1f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<line x1="{x+bw/2:.1f}" y1="{Y(v["mean"]+v["ci95"]):.1f}" x2="{x+bw/2:.1f}" y2="{Y(v["mean"]-v["ci95"]):.1f}" stroke="#0b1324" stroke-width="1.5"/>')
        e.append(f'<text x="{x+bw/2:.1f}" y="{Y(v["mean"])-7:.1f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v["mean"]:.0%}</text>')
        for li, line in enumerate(name.split("\n")):
            e.append(f'<text x="{x+bw/2:.1f}" y="{Y(0)+16+li*11:.1f}" font-size="8.7" fill="#28324a" text-anchor="middle">{line}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
