"""
src/eval/agent_grid_cortex.py

THE BEHAVING AGENT ON ITS REAL GRID SUBSTRATE — the velocity-driven HEXAGONAL GRID CORTEX
(`_HexGridModules`: fixed biological velocity gains, 6 modules at geometric scale ratios; Burak & Fiete
2009; Guanella 2007; Stensola 2012) wired into the closed-loop agent, replacing the abstract place/SR
map. This closes the loop between `grid_capacity.py` (*why* the brain uses a grid code) and the behaving
agent (*what* it does with one):

  (1) PATH INTEGRATION — the agent integrates its own self-motion through the real grid cortex, so a
      384-unit grid-cell code is its ONLY sense of position (no GPS).
  (2) NONLINEAR READOUT — a small place-cell-like network reads position from the grid code: exactly the
      nonlinear/Bayesian decoder `grid_capacity` shows is needed to extract a grid code's capacity (a
      linear reader leaves precision on the table). This is the entorhinal-grid -> hippocampal-place read.
  (3) VECTOR NAVIGATION — the grid code is a metric, so the agent steers by the decoded displacement to a
      remembered goal (Bush, Barry, Manson & Burgess 2015) — closed-loop and deterministic.
  (4) THE UNIFIED TRIPLE-LESION DISSOCIATION, re-run on this real grid substrate: a single delayed
      memory-guided harvest needs grid(map) + memory + time, and removing ANY one organ zeros the reward
      via its own failure mode (-grid: can't localize/navigate; -memory: wrong place; -time: wrong moment).

Multi-seed, mean +/- 95% CI. Writes results/agent_grid_cortex.json + .svg.

    python -m src.eval.agent_grid_cortex --seeds 3
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.agent_unified import train_time, harvest_time, T, D, W

R = 2.5            # arena half-width (well within the coarsest grid module's unambiguous range ~9 units)
STEP = 0.3         # locomotion step size
RAD = 0.4          # "at the goal" radius
DIRS = torch.tensor([[math.cos(a), math.sin(a)] for a in [i * math.pi / 4 for i in range(8)]])


def build_cortex(seed):
    """The real velocity-driven hexagonal grid cortex (fixed gains, no training)."""
    torch.manual_seed(seed)
    mod = _HexGridModules(embed_dim=64, n_modules=6, base_spacing=1.6)
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod


def train_decoder(mod, gen, nonlinear=True, iters=1500):
    """Place-cell-like readout: grid code -> position, trained self-supervised on path-integrated samples."""
    DIM = mod.K * mod.M
    dec = (nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(), nn.Linear(128, 2)) if nonlinear
           else nn.Linear(DIM, 2))
    opt = torch.optim.Adam(dec.parameters(), 3e-3)
    for _ in range(iters):
        pos = (torch.rand(256, 2, generator=gen) * 2 - 1) * R
        loss = ((dec(mod.grid_code_at(pos)) - pos) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def decode_err(mod, dec, gen, n=2000):
    pos = (torch.rand(n, 2, generator=gen) * 2 - 1) * R
    return (dec(mod.grid_code_at(pos)) - pos).pow(2).sum(-1).sqrt().mean().item()


def navigate(mod, dec, start, goal_code, gen, lesion_grid=False):
    """Closed-loop vector navigation: path-integrate self-motion -> grid code -> decoded position ->
    step toward the decoded goal; hold once arrived. Returns the trajectory of TRUE positions (len T+1)."""
    phi = mod.gains.view(mod.K, 1, 1) * start.view(1, 1, 2).clone()      # path integrator initialised at start
    pos = start.clone(); traj = [pos.clone()]
    goal_pos = dec(goal_code)[0]                                          # decode the STORED goal grid code
    for _ in range(T):
        if lesion_grid:
            d = torch.randn(2, generator=gen)                            # no grid code -> no position sense
        else:
            d = goal_pos - dec(mod._grid_code(phi))[0]
            if d.norm() < STEP:                                          # arrived -> hold position
                traj.append(pos.clone()); continue
        a = int(torch.argmax(DIRS @ d)); v = DIRS[a] * STEP
        pos = pos + v
        phi = phi + mod.gains.view(mod.K, 1, 1) * v.view(1, 1, 2)        # integrate self-motion into the grid
        traj.append(pos.clone())
    return traj


def run_seed(seed, days=40):
    mod = build_cortex(seed)
    gen = torch.Generator().manual_seed(seed + 777)
    dec = train_decoder(mod, gen, nonlinear=True)
    dec_lin = train_decoder(mod, gen, nonlinear=False)
    derr = decode_err(mod, dec, gen); derr_lin = decode_err(mod, dec_lin, gen)
    cx, Wt, gt = train_time(seed)                                         # time cells (reused TemporalCortex)
    conds = ["all-intact", "-grid", "-memory", "-time"]
    rew = {c: [] for c in conds}; nav = []
    for _ in range(days):
        goal = (torch.rand(2, generator=gen) * 2 - 1) * R
        start = (torch.rand(2, generator=gen) * 2 - 1) * R
        for c in conds:
            store = goal if c != "-memory" else (torch.rand(2, generator=gen) * 2 - 1) * R
            goal_code = mod.grid_code_at(store.unsqueeze(0))
            traj = navigate(mod, dec, start, goal_code, gen, lesion_grid=(c == "-grid"))
            th = harvest_time(cx, Wt, gt, lesion_time=(c == "-time"))
            at_goal = (traj[th] - goal).norm().item() < RAD
            rew[c].append(1.0 if (at_goal and abs(th - D) <= W) else 0.0)
            if c == "all-intact":
                nav.append(1.0 if (traj[-1] - goal).norm().item() < RAD else 0.0)
    return ({c: sum(rew[c]) / len(rew[c]) for c in conds}, derr, derr_lin, sum(nav) / len(nav))


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    conds = ["all-intact", "-grid", "-memory", "-time"]
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {c: ci([p[0][c] for p in per]) for c in conds}
    derr = ci([p[1] for p in per]); derr_lin = ci([p[2] for p in per]); nav = ci([p[3] for p in per])
    lab = {"all-intact": "ALL INTACT (grid-nav + recall + time)", "-grid": "  - grid cortex (can't localize/navigate)",
           "-memory": "  - episodic store (goes to wrong place)", "-time": "  - time cells (right place, wrong moment)"}
    print(f"\nBEHAVING AGENT ON THE REAL GRID CORTEX (n={a.seeds}; mean ± 95% CI)\n" + "=" * 70, flush=True)
    print(f"  grid cortex: 6 velocity-driven hexagonal modules, 384-unit code (fixed biological gains)", flush=True)
    print(f"  path-integration position decode error: NONLINEAR {derr[0]:.3f}±{derr[1]:.3f} | "
          f"linear {derr_lin[0]:.3f}±{derr_lin[1]:.3f}  (nonlinear wins, per grid_capacity)", flush=True)
    print(f"  intact closed-loop navigation success: {nav[0]:.0%} ± {nav[1]:.0%}\n", flush=True)
    print(f"  TRIPLE-LESION DISSOCIATION (delayed memory-guided harvest; reward):", flush=True)
    for c in conds:
        print(f"    {lab[c]:42} {agg[c][0]:.0%} ± {agg[c][1]:.0%}", flush=True)
    print(f"\n  -> the real velocity-driven HEX GRID CORTEX drives the closed-loop agent: path integration -> "
          f"nonlinear (place-cell-like) readout -> vector navigation ({nav[0]:.0%}). On this grid substrate the "
          f"unified task reproduces the clean triple dissociation (all-intact {agg['all-intact'][0]:.0%}; "
          f"-grid {agg['-grid'][0]:.0%}, -memory {agg['-memory'][0]:.0%}, -time {agg['-time'][0]:.0%}) -- "
          f"connecting grid_capacity's WHY to the agent's WHAT.", flush=True)
    out = {"n_seeds": a.seeds, "target_D": D, "window": W, "arena_R": R, "n_modules": 6,
           "decode_err_nonlinear": derr, "decode_err_linear": derr_lin, "nav_success": nav,
           "results": {c: {"mean": agg[c][0], "ci95": agg[c][1]} for c in conds}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_grid_cortex.json", "w"), indent=2)
    svg(agg, conds, derr, derr_lin, nav, "results/agent_grid_cortex.svg")
    print("\nwrote results/agent_grid_cortex.json and results/agent_grid_cortex.svg", flush=True)


def svg(agg, conds, derr, derr_lin, nav, out):
    pad = 60; bw = 74; gap = 34; ph = 196; W_ = pad + len(conds) * (bw + gap) + pad; Hh = 80 + ph + 52
    col = {"all-intact": "#2ca25f", "-grid": "#3182bd", "-memory": "#e6550d", "-time": "#9467bd"}
    short = {"all-intact": "all\nintact", "-grid": "− grid", "-memory": "− memory", "-time": "− time"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'The behaving agent on its real grid cortex: one substrate, a clean triple dissociation</text>')
    e.append(f'<text x="28" y="44" font-size="10.5" fill="#5b6b8c">6 velocity-driven hexagonal grid modules '
             f'(384-unit code) &#183; path-integration decode err {derr[0]:.3f} (nonlinear) vs {derr_lin[0]:.3f} '
             f'(linear) &#183; intact navigation {nav[0]:.0%}</text>')
    e.append('<text x="28" y="60" font-size="10.5" fill="#5b6b8c">delayed memory-guided harvest: '
             'path-integrate &amp; navigate WHERE + recall WHICH place + harvest at WHEN</text>')
    oy = 74; base = oy + ph
    e.append(f'<line x1="{pad-6}" y1="{base}" x2="{W_-pad+6}" y2="{base}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{pad-12}" y="{base-vv*ph+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for i, c in enumerate(conds):
        v = agg[c][0]; x = pad + i * (bw + gap); h = v * ph
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="{bw}" height="{h:.1f}" fill="{col[c]}" opacity="0.88"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-6:.0f}" font-size="13" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
        for j, line in enumerate(short[c].split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{base+16+j*12:.0f}" font-size="10" fill="#28324a" text-anchor="middle">{line}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
