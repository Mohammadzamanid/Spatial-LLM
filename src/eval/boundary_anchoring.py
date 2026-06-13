"""
src/eval/boundary_anchoring.py

Toward the real brain: do environmental BOUNDARIES correct path-integration drift?

Real path integration is NOISY, so grid/place estimates DRIFT as you travel (error grows
with distance). The brain corrects this by re-anchoring grid phase to environmental
boundaries (Hardcastle, Ganguli & Giocomo 2015 — boundaries are a grid error-correction
mechanism). We test the same in our velocity-driven grid cortex.

Setup: an agent does bounded random walks in a square arena (walls at ±R). The grid modules
integrate NOISY velocity (-> drift). With boundary anchoring on, boundary-vector cells read
the (distance, bearing) to the nearest wall and gate-reset the grid phase. We decode position
from the grid rep and measure error vs path length T, for three conditions:

  exact     : no integration noise, no anchor   (drift-free floor)
  drift     : noisy integration,    no anchor   (error grows with T)
  anchored  : noisy integration,    + boundaries (error should stay BOUNDED)

Writes results/boundary_anchoring.json and results/boundary_anchoring.svg.
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import _HexGridModules


def bounded_walks(n, T, R=3.0, speed=(0.2, 0.8), seed=0):
    """Random walks that bounce off the walls of a [-R,R]^2 arena. Returns the ACTUAL
    per-step velocity v3d (B,T,3; vz=0), the boundary observation (dist,bearing to nearest
    wall) per step, and the final (x,y)."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.zeros(n, 2)
    v3d = torch.zeros(n, T, 3); bobs = torch.zeros(n, T, 2)
    tpos = torch.zeros(n, T, 2); amask = torch.zeros(n, T, 2)     # true position + constrained-axis mask
    wall_bearing = torch.tensor([0.0, math.pi, math.pi / 2, -math.pi / 2])   # right,left,top,bottom
    for t in range(T):
        heading = torch.rand(n, generator=g) * 2 * math.pi
        sp = torch.rand(n, generator=g) * (speed[1] - speed[0]) + speed[0]
        step = torch.stack([sp * heading.cos(), sp * heading.sin()], -1)
        newpos = (pos + step).clamp(-R, R)                 # walls stop you
        v3d[:, t, :2] = newpos - pos                        # actual displacement (clipped at walls)
        pos = newpos
        dists = torch.stack([R - pos[:, 0], pos[:, 0] + R, R - pos[:, 1], pos[:, 1] + R], -1)  # (n,4)
        dmin, which = dists.min(-1)
        bobs[:, t, 0] = dmin
        bobs[:, t, 1] = wall_bearing[which]
        tpos[:, t] = pos
        amask[:, t, 0] = (which < 2).float()                # left/right wall -> constrains x
        amask[:, t, 1] = (which >= 2).float()               # top/bottom wall -> constrains y
    return v3d, bobs, pos, tpos, amask


def run(cond, R=3.0, epochs=70, n=4000, Ttr=(8, 12, 16, 20), Tev=(6, 12, 18, 24, 30),
        noise=0.12, seed=0):
    noise_std = 0.0 if cond == "exact" else noise
    anchor = cond in ("anchored", "anchored_learned")
    learned = (cond == "anchored_learned")
    torch.manual_seed(seed)
    # more modules + coarser base spacing -> the population code is unique over the arena;
    # a nonlinear decoder is needed to invert the periodic (residue) grid code.
    cx = _HexGridModules(embed_dim=64, n_modules=5, base_spacing=1.5,
                         noise_std=noise_std, boundary_anchor=anchor, learned_loc=learned)
    decoder = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
    opt = torch.optim.Adam(list(cx.parameters()) + list(decoder.parameters()), lr=3e-3)
    mse = nn.MSELoss()
    cx.train()
    for ep in range(epochs):
        T = Ttr[ep % len(Ttr)]
        v3d, bobs, pos, tpos, amask = bounded_walks(n, T, R, seed=1000 + ep)
        opt.zero_grad()
        _, gc = cx(v3d, boundary_obs=bobs if anchor else None, return_cells=True)
        loss = mse(decoder(gc), pos)
        if learned:
            # EXPERIENTIAL supervision: learn to localize from boundary cells (no arena geometry).
            # The boundary head predicts the position it can sense (constrained axis) and learns to
            # TRUST it near walls — replacing the hard-coded R-dist fix.
            B2, T2 = v3d.shape[0], v3d.shape[1]
            dist = bobs[:, :, 0].reshape(-1); bearing = bobs[:, :, 1].reshape(-1)
            emb = cx.bvc(dist, bearing)
            p_hat = cx.bvc_to_pos(emb).view(B2, T2, 2)
            gate = torch.sigmoid(cx.bvc_gate(emb)).view(B2, T2, 2)
            prox = torch.exp(-bobs[:, :, 0:1] / cx.anchor_scale)            # (B,T,1)
            aux = (((p_hat - tpos) ** 2) * amask).sum() / (amask.sum() + 1e-6)     # localize constrained axis
            aux = aux + ((gate - amask * prox) ** 2).mean()                # trust high near walls
            loss = loss + 0.5 * aux
        loss.backward()
        opt.step()
    cx.eval()
    out = {}
    with torch.no_grad():
        for T in Tev:
            v3d, bobs, pos, tpos, amask = bounded_walks(3000, T, R, seed=5000 + T)
            errs = []
            for r in range(3):                                 # average over noise draws
                _, gc = cx(v3d, boundary_obs=bobs if anchor else None, return_cells=True)
                errs.append((decoder(gc) - pos).norm(dim=1).mean().item())
            out[T] = round(sum(errs) / len(errs), 4)
    return out


# ------------------------------------------------------------------------- svg plot
def plot_svg(curves, Tev, out="results/boundary_anchoring.svg"):
    W, H, pad = 640, 420, 60
    xmax = max(Tev); ymax = max(max(c.values()) for c in curves.values()) * 1.1 + 1e-6
    def X(t): return pad + (t - min(Tev)) / (xmax - min(Tev)) * (W - 2 * pad)
    def Y(v): return H - pad - v / ymax * (H - 2 * pad)
    col = {"exact": "#2ca25f", "drift": "#de2d26", "anchored": "#3b528b",
           "anchored_learned": "#8856a7"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="40" y="30" font-size="17" font-weight="800" fill="#0b1324">'
             'Boundary anchoring corrects path-integration drift</text>')
    e.append(f'<text x="40" y="50" font-size="12" fill="#5b6b8c">position-decode error vs path '
             f'length T — noisy grid integration, with/without boundary phase-reset</text>')
    e.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#33415c"/>')
    e.append(f'<text x="{W/2}" y="{H-18}" font-size="12" fill="#28324a" text-anchor="middle">path length T (steps)</text>')
    e.append(f'<text x="20" y="{H/2}" font-size="12" fill="#28324a" transform="rotate(-90 20 {H/2})" text-anchor="middle">mean position error</text>')
    for t in Tev:
        e.append(f'<text x="{X(t)}" y="{H-pad+16}" font-size="10" fill="#5b6b8c" text-anchor="middle">{t}</text>')
    for name, c in curves.items():
        pts = " ".join(f"{X(t):.1f},{Y(c[t]):.1f}" for t in Tev)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col[name]}" stroke-width="2.5"/>')
        for t in Tev:
            e.append(f'<circle cx="{X(t):.1f}" cy="{Y(c[t]):.1f}" r="3" fill="{col[name]}"/>')
    ly = pad + 6
    for name in curves:
        e.append(f'<rect x="{W-pad-180}" y="{ly}" width="14" height="4" fill="{col[name]}"/>')
        e.append(f'<text x="{W-pad-160}" y="{ly+6}" font-size="12" fill="#28324a">{name}</text>')
        ly += 20
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--epochs", type=int, default=70)
    a = ap.parse_args()
    Tev = (6, 12, 18, 24, 30)
    curves = {}
    for cond in ("exact", "drift", "anchored", "anchored_learned"):
        curves[cond] = run(cond, noise=a.noise, epochs=a.epochs, Tev=Tev)
        flat = "  ".join(f"T{t}:{curves[cond][t]:.3f}" for t in Tev)
        print(f"[{cond:16}] position error by length:  {flat}", flush=True)
    svg = plot_svg(curves, Tev)
    tm = max(Tev); drift_end = curves["drift"][tm]
    def red(name):
        return round(100 * (drift_end - curves[name][tm]) / max(drift_end, 1e-9), 1)
    print(f"\nat T={tm}: drift={drift_end:.3f}  "
          f"anchored(geometric)={curves['anchored'][tm]:.3f} (-{red('anchored')}%)  "
          f"anchored(LEARNED)={curves['anchored_learned'][tm]:.3f} (-{red('anchored_learned')}%)", flush=True)
    with open("results/boundary_anchoring.json", "w") as f:
        json.dump({"noise": a.noise,
                   "curves": {k: {str(t): v for t, v in c.items()} for k, c in curves.items()},
                   "drift_reduction_pct_at_Tmax": {n: red(n) for n in ("anchored", "anchored_learned")}},
                  f, indent=2)
    print(f"wrote results/boundary_anchoring.json and {svg}", flush=True)


if __name__ == "__main__":
    main()
