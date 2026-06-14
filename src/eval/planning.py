"""
src/eval/planning.py

PLANNING — the cognitive map as a PLANNER, not just a recorder (Tolman's shortcut test).

The grid code is a linear metric (phase = gain * position), so the displacement between any two
remembered places is just the difference of their grid codes — a VECTOR the agent can read off
even for a route it never walked (vector navigation: Bush et al. 2015; Banino et al. 2018). And it
can FORWARD-REPLAY that route: sweep the grid phase from here to the goal to imagine the path before
moving (preplay: Pfeiffer & Foster 2013).

Test: the agent reaches A and B by two SEPARATE winding walks from home (it never travels A->B).
From the map it then plans the direct A->B shortcut. We measure how well the planned vector matches
the true one, whether it is navigable, the forward-replay sweep, and how much shorter the shortcut is
than retracing the known routes (via home).

Writes results/planning.json and results/planning.svg (winding paths + planned shortcuts).
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


def winding_walk(n, T, R, seed):
    """n random walks from home (0,0); returns the full paths (n,T+1,2) and endpoints (n,2)."""
    g = torch.Generator().manual_seed(seed); pos = torch.zeros(n, 2); path = [pos.clone()]
    for _ in range(T):
        h = torch.rand(n, generator=g) * 2 * math.pi; s = torch.rand(n, generator=g) * 0.6 + 0.2
        pos = (pos + torch.stack([s * h.cos(), s * h.sin()], -1)).clamp(-R, R)
        path.append(pos.clone())
    return torch.stack(path, 1), pos


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--R", type=float, default=3.0); a = ap.parse_args()
    R = a.R; torch.manual_seed(0)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)            # the grid metric

    # the map readout: grid code -> position (the cognitive map, learned once)
    posT = (torch.rand(8000, 2) * 2 - 1) * R
    dec = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(800):
        opt.zero_grad(); F.mse_loss(dec(grid_code(cx, posT)), posT).backward(); opt.step()

    # the agent reaches A and B by two SEPARATE winding walks from home; it never travels A->B
    pathA, A = winding_walk(3000, 14, R, 1)
    pathB, B = winding_walk(3000, 14, R, 2)
    with torch.no_grad():
        Ahat = dec(grid_code(cx, A)); Bhat = dec(grid_code(cx, B))    # map readouts of A, B
        planned = Bhat - Ahat                                        # the PLANNED A->B shortcut vector
    true = B - A

    cos = F.cosine_similarity(planned, true, dim=1).clamp(-1, 1)
    dir_err = torch.rad2deg(torch.acos(cos))
    dist_err = (planned.norm(-1) - true.norm(-1)).abs() / (true.norm(-1) + 1e-6)
    navigable = (dir_err < 15).float().mean().item()
    # Tolman: the shortcut |A-B| vs RETRACING known routes via home (|A|+|B|)
    short = true.norm(-1); detour = A.norm(-1) + B.norm(-1)
    savings = (1 - short / (detour + 1e-6)).clamp(min=0).mean().item()

    # forward replay (preplay): sweep the grid phase A->B, decode each -> the imagined path; how far
    # does it deviate from the straight line to the goal?
    with torch.no_grad():
        devs = []
        for t in torch.linspace(0, 1, 11):
            p = Ahat + t * (Bhat - Ahat)                             # imagined position along the plan
            d = dec(grid_code(cx, p))                                # decode the swept grid code
            seg = (Bhat - Ahat); segn = seg / (seg.norm(-1, keepdim=True) + 1e-6)
            perp = (d - Ahat) - ((d - Ahat) * segn).sum(-1, keepdim=True) * segn
            devs.append(perp.norm(-1))
        replay_dev = torch.stack(devs).mean().item()

    out = {"n_pairs": A.shape[0], "shortcut_dir_error_deg_mean": round(dir_err.mean().item(), 2),
           "shortcut_dir_error_deg_median": round(dir_err.median().item(), 2),
           "shortcut_dist_rel_error": round(dist_err.mean().item(), 3),
           "frac_navigable(<15deg)": round(navigable, 3),
           "forward_replay_line_deviation": round(replay_dev, 3),
           "detour_savings_vs_retrace_home": round(savings, 3)}
    print("PLANNING — Tolman shortcut from the cognitive map (grid-code vector navigation):", flush=True)
    print(f"  planned A->B shortcut: direction error {out['shortcut_dir_error_deg_mean']}° mean "
          f"({out['shortcut_dir_error_deg_median']}° median), distance rel-err "
          f"{out['shortcut_dist_rel_error']}", flush=True)
    print(f"  {100*navigable:.0f}% of shortcuts are navigable (<15° off) — routes NEVER travelled", flush=True)
    print(f"  forward-replay sweep stays {out['forward_replay_line_deviation']} off the straight line "
          f"(coherent imagined path to the goal)", flush=True)
    print(f"  the shortcut is {100*savings:.0f}% shorter than retracing the known routes via home", flush=True)

    # SVG: home, two winding experienced paths, and the planned straight shortcut (a few examples)
    sep = true.norm(-1); idx = sep.argsort(descending=True)[:3]
    svg_plan([(pathA[i], pathB[i], A[i], B[i]) for i in idx.tolist()], R, "results/planning.svg")
    os.makedirs("results", exist_ok=True)
    with open("results/planning.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/planning.json and results/planning.svg", flush=True)


def svg_plan(examples, R, out):
    n = len(examples); pad = 16; sz = 200; W = pad + n * (sz + pad); H = sz + 56
    def X(x, off): return off + (x + R) / (2 * R) * sz
    def Y(y): return 44 + (R - y) / (2 * R) * sz
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="16" y="26" font-size="16" font-weight="800" fill="#0b1324">'
             'Planning a NOVEL shortcut from the cognitive map (Tolman test)</text>')
    for k, (pa, pb, A, B) in enumerate(examples):
        off = pad + k * (sz + pad)
        e.append(f'<rect x="{off}" y="44" width="{sz}" height="{sz}" fill="#f4f7fb" stroke="#33415c"/>')
        for path, col in [(pa, "#9aa5b8"), (pb, "#9aa5b8")]:                 # winding experienced routes
            pts = " ".join(f"{X(p[0].item(),off):.1f},{Y(p[1].item()):.1f}" for p in path)
            e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
        e.append(f'<line x1="{X(A[0].item(),off):.1f}" y1="{Y(A[1].item()):.1f}" '                # the PLAN
                 f'x2="{X(B[0].item(),off):.1f}" y2="{Y(B[1].item()):.1f}" stroke="#e6550d" '
                 f'stroke-width="2.6" stroke-dasharray="5,3"/>')
        e.append(f'<circle cx="{X(0,off):.1f}" cy="{Y(0):.1f}" r="4" fill="#2ca25f"/>')           # home
        for P, c, lab in [(A, "#3b528b", "A"), (B, "#3b528b", "B")]:
            e.append(f'<circle cx="{X(P[0].item(),off):.1f}" cy="{Y(P[1].item()):.1f}" r="4" fill="{c}"/>')
            e.append(f'<text x="{X(P[0].item(),off)+6:.1f}" y="{Y(P[1].item())-6:.1f}" font-size="12" fill="#28324a">{lab}</text>')
    e.append(f'<text x="16" y="{H-6}" font-size="11" fill="#5b6b8c">grey = winding routes actually '
             f'walked (home→A, home→B); orange dashed = planned A→B shortcut, never travelled; green = home</text>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
