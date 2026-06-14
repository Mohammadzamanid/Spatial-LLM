"""
src/eval/continual.py

ONE-SHOT & CONTINUAL LEARNING — instant place fields, no catastrophic forgetting.

The hippocampus encodes a place in ONE visit (behavioural-timescale plasticity; Bittner & Magee
2017) and stores many memories without overwriting old ones — pattern-separated, expandable, local.
This is the fast hippocampal half of Complementary Learning Systems (McClelland, McNaughton &
O'Reilly 1995): a single gradient-trained network learning places one-at-a-time CATASTROPHICALLY
FORGETS; binding each place one-shot to its grid code does not.

We bind each visited location, in ONE local write, to a place cell w = grid-code(L). Tests:
  1. ONE-SHOT place field — after a single visit, the cell fires selectively at L (a localized field).
  2. CONTINUAL, no forgetting — visit K locations sequentially (one-shot each); afterwards every one is
     still recalled. A shared classifier trained the same sequence by gradient forgets the early ones.

Writes results/continual.json and results/continual.svg.
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
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--K", type=int, default=20)        # locations learned sequentially
    a = ap.parse_args()
    R, K = a.R, a.K
    torch.manual_seed(0)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    locs = (torch.rand(K, 2) * 2 - 1) * R                              # K places, visited in order 0..K-1

    def nrm(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    # ---- ONE-SHOT Hebbian place memory: each visit writes one place cell w = grid-code(L) ----
    W = []                                                            # grows as places are visited
    for i in range(K):
        W.append(nrm(grid_code(cx, locs[i:i + 1]))[0])               # ONE local write per visit
    W = torch.stack(W)                                               # (K, griddim)

    @torch.no_grad()
    def hebbian_recall(codes):                                       # which stored place is this?
        return (nrm(codes) @ W.t()).argmax(1)

    # ---- BASELINE: one shared classifier trained the SAME sequence by gradient (no hippocampus) ----
    clf = nn.Sequential(nn.Linear(cx.K * cx.M, 128), nn.ReLU(), nn.Linear(128, K))
    opt = torch.optim.Adam(clf.parameters(), lr=5e-3)
    for i in range(K):                                              # visit places one at a time
        for _ in range(30):
            x = grid_code(cx, locs[i] + 0.15 * torch.randn(64, 2)); y = torch.full((64,), i)
            opt.zero_grad(); F.cross_entropy(clf(x), y).backward(); opt.step()

    # ---- recall of every place AFTER all K learned (revisit near each, a few noisy probes) ----
    @torch.no_grad()
    def recall_by_order(predict, trials=16):
        acc = torch.zeros(K)
        for _ in range(trials):
            probe = grid_code(cx, locs + 0.15 * torch.randn(K, 2))
            acc += (predict(probe) == torch.arange(K)).float()
        return acc / trials
    heb = recall_by_order(hebbian_recall)
    sgd = recall_by_order(lambda c: clf(c).argmax(1))

    # one-shot place field: rate map of the FIRST cell (formed from a single visit)
    Gn = 32; xs = torch.linspace(-R, R, Gn)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    gridpos = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    with torch.no_grad():
        field = (nrm(grid_code(cx, gridpos)) @ W[0]).reshape(Gn, Gn)
    # field localization: fraction of arena above half-max (small = sharp single field)
    fn = (field - field.min()) / (field.max() - field.min() + 1e-9)
    field_area = (fn > 0.5).float().mean().item()

    # bin recall by learning age (oldest 25% ... newest 25%)
    q = K // 4
    def bins(v): return [round(v[j * q:(j + 1) * q].mean().item(), 3) for j in range(4)]
    out = {"K": K, "hebbian_recall_mean": round(heb.mean().item(), 3),
           "sgd_recall_mean": round(sgd.mean().item(), 3),
           "hebbian_recall_by_age_quartile_oldest_to_newest": bins(heb),
           "sgd_recall_by_age_quartile_oldest_to_newest": bins(sgd),
           "oneshot_place_field_area": round(field_area, 3)}
    print("ONE-SHOT & CONTINUAL LEARNING (hippocampal one-shot vs gradient forgetting):", flush=True)
    print(f"  ONE-SHOT place field from a single visit: area {field_area:.2f} of arena (localized field)", flush=True)
    print(f"  recall of all {K} places after sequential learning:  "
          f"one-shot Hebbian {100*heb.mean():.0f}%  vs  gradient baseline {100*sgd.mean():.0f}%", flush=True)
    print(f"  recall by age (oldest->newest quartile):  Hebbian {bins(heb)}  |  gradient {bins(sgd)} "
          f"(gradient FORGETS the old ones)", flush=True)

    svg_continual(field, R, heb, sgd, "results/continual.svg")
    os.makedirs("results", exist_ok=True)
    with open("results/continual.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/continual.json and results/continual.svg", flush=True)


def _cmap(v):
    st = [(0.0, (68, 1, 84)), (0.5, (33, 144, 141)), (1.0, (253, 231, 37))]
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(st) - 1):
        x, y = st[i], st[i + 1]
        if v <= y[0]:
            f = (v - x[0]) / (y[0] - x[0] + 1e-9)
            c = [round(x[1][k] + f * (y[1][k] - x[1][k])) for k in range(3)]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
    return "#fde725"


def svg_continual(field, R, heb, sgd, out):
    Gn = field.shape[0]; cell = 200 / Gn; pad = 20
    px = pad; py = 56                                                 # left panel: place field
    qx = px + 200 + 70; qw = 300; qh = 200                            # right panel: recall curve
    W = qx + qw + pad; H = 300
    fn = (field - field.min()) / (field.max() - field.min() + 1e-9)
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'One-shot place fields &amp; continual learning without forgetting</text>')
    e.append(f'<text x="{px}" y="50" font-size="11.5" fill="#28324a">place field from a SINGLE visit</text>')
    for i in range(Gn):
        for j in range(Gn):
            e.append(f'<rect x="{px+i*cell:.1f}" y="{py+(Gn-1-j)*cell:.1f}" width="{cell+0.6:.1f}" '
                     f'height="{cell+0.6:.1f}" fill="{_cmap(fn[i, j].item())}"/>')
    # right: recall vs learning age
    K = len(heb)
    def X(i): return qx + i / (K - 1) * qw
    def Y(v): return py + qh - v * qh
    e.append(f'<text x="{qx}" y="50" font-size="11.5" fill="#28324a">recall vs learning age (old→new)</text>')
    e.append(f'<line x1="{qx}" y1="{py+qh}" x2="{qx+qw}" y2="{py+qh}" stroke="#33415c"/>'
             f'<line x1="{qx}" y1="{py}" x2="{qx}" y2="{py+qh}" stroke="#33415c"/>')
    for v in (0.0, 0.5, 1.0):
        e.append(f'<text x="{qx-26}" y="{Y(v)+4:.1f}" font-size="9" fill="#5b6b8c">{int(v*100)}%</text>')
    for series, col, lab in [(heb, "#2ca25f", "one-shot Hebbian"), (sgd, "#de2d26", "gradient (shared)")]:
        pts = " ".join(f"{X(i):.1f},{Y(series[i].item()):.1f}" for i in range(K))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    ly = py + 8
    for col, lab in [("#2ca25f", "one-shot Hebbian (no forgetting)"), ("#de2d26", "gradient baseline (forgets old)")]:
        e.append(f'<rect x="{qx+qw-185}" y="{ly}" width="13" height="4" fill="{col}"/>')
        e.append(f'<text x="{qx+qw-168}" y="{ly+6}" font-size="10.5" fill="#28324a">{lab}</text>'); ly += 18
    e.append(f'<text x="{qx}" y="{py+qh+22}" font-size="10" fill="#5b6b8c">learning order (oldest place … newest)</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
