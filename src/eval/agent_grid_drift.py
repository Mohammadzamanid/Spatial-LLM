"""
src/eval/agent_grid_drift.py

PATH-INTEGRATION DRIFT, AND ITS CORRECTION BY BOUNDARY-VECTOR CELLS (the Fiete caveat, resolved).

A grid code path-integrates self-motion, but real self-motion is NOISY, so the integrated grid phase
DRIFTS away from the true position — error that accumulates without bound (Burak & Fiete 2009; the famous
caveat to grid path integration). The brain corrects this with ALLOTHETIC cues: when the animal senses a
known boundary, boundary-vector cells supply an external position fix that RESETS the accumulated grid
error (Hardcastle, Ganguli & Giocomo 2015; Etienne & Jeffery; Knierim). We reproduce both halves on the
closed-loop grid-cortex agent (`agent_grid_cortex.py`), using the REAL `BoundaryVectorCells` organ:

  (A) MECHANISM — self-localization error over a long exploratory walk. Under noisy self-motion the grid
      estimate's error GROWS UNBOUNDED (path-integration drift). Routed through boundary-vector cells —
      sense the nearest wall -> BVC population -> a LEARNED allothetic read-out -> reset the grid phase,
      gated by wall proximity — the error stays BOUNDED (the stationary sawtooth: drift, then reset at a
      wall). Nothing is hard-coded: the allothetic localizer is learned self-supervised from the BVC code.

  (B) BEHAVIOR — foraging (visit a sequence of goals over a long episode). Drift COMPOUNDS over the
      episode, so without anchoring the agent reaches fewer and fewer goals as noise grows; BVC anchoring
      keeps the grid estimate calibrated and substantially rescues foraging.

Multi-seed, mean +/- 95% CI. Writes results/agent_grid_drift.json + .svg.

    python -m src.eval.agent_grid_drift --seeds 3
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.spatial_cells import BoundaryVectorCells
from src.eval.agent_grid_cortex import build_cortex, train_decoder, R, STEP, DIRS

NOISES_A = [0.0, 0.05, 0.10, 0.15]          # self-motion noise levels for the localization walk
TRACE_NOISE = 0.10                          # noise level whose error-vs-time trace we plot
NOISES_B = [0.0, 0.05, 0.10, 0.15, 0.20]    # noise levels for foraging
WALK_STEPS = 120; N_WALKS = 24
N_GOALS = 6; GOAL_STEPS = 30; FORAGE_EPISODES = 80
ANCHOR_SCALE = 0.35                         # boundary-proximity gate width (cells fire only near a wall)


def sense_wall(pos):
    """Allothetic boundary sense: nearest wall -> (distance, allocentric bearing, axis, signed coord)."""
    x, y = pos[0].item(), pos[1].item()
    dx, dy = R - abs(x), R - abs(y)
    if dx <= dy:
        return dx, (0.0 if x > 0 else math.pi), 0, x
    return dy, (math.pi / 2 if y > 0 else -math.pi / 2), 1, y


def train_bvc(gen, iters=1500):
    """Real BoundaryVectorCells organ + a small LEARNED read-out: BVC(dist,bearing) -> wall coordinate.
    Trained self-supervised (near-wall weighted) — the allothetic localizer is learned, not a formula."""
    bvc = BoundaryVectorCells(num_cells=24, embed_dim=32, max_distance=R)
    loc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(list(bvc.parameters()) + list(loc.parameters()), 3e-3)
    for _ in range(iters):
        pos = (torch.rand(256, 2, generator=gen) * 2 - 1) * R
        sw = [sense_wall(p) for p in pos]
        dist = torch.tensor([s[0] for s in sw]); bear = torch.tensor([s[1] for s in sw])
        perp = torch.tensor([s[3] for s in sw]).unsqueeze(1)
        loss = ((loc(bvc(dist, bear)) - perp) ** 2 * torch.exp(-dist / 0.5).unsqueeze(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in list(bvc.parameters()) + list(loc.parameters()):
        p.requires_grad_(False)
    return bvc, loc


def bvc_coord_err(bvc, loc, gen, n=400):
    pos = (torch.rand(n, 2, generator=gen) * 2 - 1) * R
    sw = [sense_wall(p) for p in pos]
    dd = torch.tensor([s[0] for s in sw]); bb = torch.tensor([s[1] for s in sw]); pp = torch.tensor([s[3] for s in sw])
    near = dd < 0.4
    return (loc(bvc(dd, bb))[:, 0] - pp)[near].abs().mean().item()


def anchor(phi, pos, bvc, loc, gains):
    """Boundary reset: sense the nearest wall, decode the allothetic coordinate from the BVC organ, and
    pull that axis's grid phase toward it — gated by wall proximity (strong only near a boundary)."""
    dist, bear, axis, _ = sense_wall(pos)
    w = math.exp(-dist / ANCHOR_SCALE)
    perp = loc(bvc(torch.tensor([dist]), torch.tensor([bear])))[0, 0]
    phi[:, 0, axis] = (1 - w) * phi[:, 0, axis] + w * (gains * perp)
    return phi


def walk(mod, dec, bvc, loc, gen, noise, do_anchor):
    """Long exploratory (momentum) walk; return the self-localization error at each step."""
    gains = mod.gains
    pos = (torch.rand(2, generator=gen) * 2 - 1) * R * 0.5
    phi = gains.view(mod.K, 1, 1) * pos.view(1, 1, 2).clone()
    heading = torch.rand(1, generator=gen) * 2 * math.pi
    errs = []
    for _ in range(WALK_STEPS):
        heading = heading + torch.randn(1, generator=gen) * 0.5
        v = torch.tensor([math.cos(heading.item()), math.sin(heading.item())]) * STEP
        if abs((pos + v)[0]) > R or abs((pos + v)[1]) > R:
            heading = heading + math.pi; v = -v
        pos = (pos + v).clamp(-R, R)
        phi = phi + gains.view(mod.K, 1, 1) * (v + torch.randn(2, generator=gen) * noise).view(1, 1, 2)
        if do_anchor:
            phi = anchor(phi, pos, bvc, loc, gains)
        errs.append((dec(mod._grid_code(phi))[0] - pos).norm().item())
    return errs


def forage(mod, dec, bvc, loc, gen, noise, do_anchor):
    """Visit N_GOALS sequential goals over one long episode; return the fraction reached."""
    gains = mod.gains
    pos = (torch.rand(2, generator=gen) * 2 - 1) * R
    phi = gains.view(mod.K, 1, 1) * pos.view(1, 1, 2).clone(); reached = 0
    for _ in range(N_GOALS):
        goal = (torch.rand(2, generator=gen) * 2 - 1) * R
        gp = dec(mod.grid_code_at(goal.unsqueeze(0)))[0]
        for _ in range(GOAL_STEPS):
            d = gp - dec(mod._grid_code(phi))[0]
            if d.norm() < STEP:
                break
            a = int(torch.argmax(DIRS @ d)); v = DIRS[a] * STEP
            pos = (pos + v).clamp(-R, R)
            phi = phi + gains.view(mod.K, 1, 1) * (v + torch.randn(2, generator=gen) * noise).view(1, 1, 2)
            if do_anchor:
                phi = anchor(phi, pos, bvc, loc, gains)
        if (pos - goal).norm().item() < 0.4:
            reached += 1
    return reached / N_GOALS


def run_seed(seed):
    mod = build_cortex(seed)
    gen = torch.Generator().manual_seed(seed + 4242)
    dec = train_decoder(mod, gen, nonlinear=True, iters=1200)
    bvc, loc = train_bvc(gen)
    cerr = bvc_coord_err(bvc, loc, gen)

    locA = {}; trace = {"na": [0.0] * WALK_STEPS, "an": [0.0] * WALK_STEPS}
    for noise in NOISES_A:
        na, an, naf, anf = [], [], [], []
        for _ in range(N_WALKS):
            e0 = walk(mod, dec, bvc, loc, gen, noise, False)
            e1 = walk(mod, dec, bvc, loc, gen, noise, True)
            na.append(sum(e0) / len(e0)); an.append(sum(e1) / len(e1))
            naf.append(sum(e0[-20:]) / 20); anf.append(sum(e1[-20:]) / 20)
            if noise == TRACE_NOISE:
                for t in range(WALK_STEPS):
                    trace["na"][t] += e0[t] / N_WALKS; trace["an"][t] += e1[t] / N_WALKS
        locA[noise] = {"na": sum(na) / len(na), "an": sum(an) / len(an),
                       "na_final": sum(naf) / len(naf), "an_final": sum(anf) / len(anf)}

    forageB = {}
    for noise in NOISES_B:
        na = sum(forage(mod, dec, bvc, loc, gen, noise, False) for _ in range(FORAGE_EPISODES)) / FORAGE_EPISODES
        an = sum(forage(mod, dec, bvc, loc, gen, noise, True) for _ in range(FORAGE_EPISODES)) / FORAGE_EPISODES
        forageB[noise] = {"na": na, "an": an}
    return {"loc": locA, "trace": trace, "forage": forageB, "bvc_err": cerr}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    locA = {nz: {k: ci([p["loc"][nz][k] for p in per]) for k in ("na", "an", "na_final", "an_final")} for nz in NOISES_A}
    forageB = {nz: {k: ci([p["forage"][nz][k] for p in per]) for k in ("na", "an")} for nz in NOISES_B}
    trace = {k: [sum(p["trace"][k][t] for p in per) / a.seeds for t in range(WALK_STEPS)] for k in ("na", "an")}
    bvc_err = ci([p["bvc_err"] for p in per])

    print(f"\nPATH-INTEGRATION DRIFT + BOUNDARY-VECTOR-CELL CORRECTION (n={a.seeds}; mean ± 95% CI)\n" + "=" * 80, flush=True)
    print(f"  BVC learned allothetic read-out: near-wall coordinate error {bvc_err[0]:.3f} ± {bvc_err[1]:.3f}\n", flush=True)
    print("(A) self-localization error over a 120-step walk (lower = better):", flush=True)
    print(f"    {'noise':>6} | {'NO anchor (mean / final)':>26} | {'BVC anchor (mean / final)':>26}", flush=True)
    for nz in NOISES_A:
        d = locA[nz]
        print(f"    {nz:>6.2f} | {d['na'][0]:>10.3f} / {d['na_final'][0]:>10.3f} | "
              f"{d['an'][0]:>10.3f} / {d['an_final'][0]:>10.3f}", flush=True)
    print("\n(B) foraging — fraction of 6 sequential goals reached:", flush=True)
    print(f"    {'noise':>6} | {'NO anchor':>10} | {'BVC anchor':>11}", flush=True)
    for nz in NOISES_B:
        d = forageB[nz]
        print(f"    {nz:>6.2f} | {d['na'][0]:>9.0%} | {d['an'][0]:>10.0%}", flush=True)
    hi = NOISES_A[-1]
    print(f"\n  -> (A) under noisy self-motion the grid estimate DRIFTS unbounded (error grows: mean "
          f"{locA[hi]['na'][0]:.2f} but final {locA[hi]['na_final'][0]:.2f} at noise {hi}); routing the boundary "
          f"sense through BOUNDARY-VECTOR CELLS bounds it ({locA[hi]['an'][0]:.2f} mean ≈ {locA[hi]['an_final'][0]:.2f} "
          f"final — the stationary sawtooth). (B) drift compounds over foraging (no-anchor "
          f"{forageB[0.2]['na'][0]:.0%} at noise 0.2) and BVC anchoring rescues it ({forageB[0.2]['an'][0]:.0%}). "
          f"The Fiete caveat and its Hardcastle-2015 correction, on the closed-loop agent.", flush=True)

    out = {"n_seeds": a.seeds, "trace_noise": TRACE_NOISE, "bvc_coord_err": bvc_err,
           "localization": {str(nz): locA[nz] for nz in NOISES_A},
           "foraging": {str(nz): forageB[nz] for nz in NOISES_B}, "trace": trace}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_grid_drift.json", "w"), indent=2)
    svg(locA, forageB, trace, bvc_err, "results/agent_grid_drift.svg")
    print("\nwrote results/agent_grid_drift.json and results/agent_grid_drift.svg", flush=True)


def svg(locA, forageB, trace, bvc_err, out):
    pad = 56; pw = 320; ph = 200; gap = 92; W = pad + 2 * pw + gap + 24; H = 80 + ph + 46
    cna, can = "#c9341a", "#2ca25f"
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Path-integration drift, corrected by boundary-vector cells (Hardcastle 2015)</text>')
    e.append(f'<text x="26" y="42" font-size="10.5" fill="#5b6b8c">noisy self-motion drifts the grid '
             f'estimate; the BVC organ (learned allothetic read-out, near-wall error {bvc_err[0]:.3f}) resets it near walls</text>')
    # Panel A: localization error vs time at TRACE_NOISE
    oxA = pad; oy = 62
    tmax = max(max(trace["na"]), max(trace["an"])) * 1.12 + 1e-6
    def XA(t): return oxA + (t / (WALK_STEPS - 1)) * pw
    def YA(v): return oy + ph - (v / tmax) * ph
    e.append(f'<text x="{oxA}" y="{oy-6}" font-size="11.5" font-weight="700" fill="#0b1324">'
             f'(A) self-localization error vs time (noise {TRACE_NOISE})</text>')
    e.append(f'<line x1="{oxA}" y1="{oy+ph}" x2="{oxA+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{oxA}" y1="{oy}" x2="{oxA}" y2="{oy+ph}" stroke="#33415c"/>')
    for who, c in (("na", cna), ("an", can)):
        pts = " ".join(f"{XA(t):.1f},{YA(trace[who][t]):.1f}" for t in range(WALK_STEPS))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>')
    e.append(f'<text x="{oxA+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">step &#8594;</text>')
    e.append(f'<text x="{oxA+6}" y="{oy+12}" font-size="9" fill="#7787a6">grows unbounded (drift) vs bounded (anchored)</text>')
    e.append(f'<text x="{oxA-8}" y="{YA(tmax/tmax*tmax)+0:.0f}" font-size="8" fill="#5b6b8c"></text>')
    # Panel B: foraging success vs noise
    oxB = pad + pw + gap
    def XB(i): return oxB + (i / (len(NOISES_B) - 1)) * pw
    def YB(v): return oy + ph - v * ph
    e.append(f'<text x="{oxB}" y="{oy-6}" font-size="11.5" font-weight="700" fill="#0b1324">'
             f'(B) foraging success vs self-motion noise</text>')
    e.append(f'<line x1="{oxB}" y1="{oy+ph}" x2="{oxB+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{oxB}" y1="{oy}" x2="{oxB}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{oxB-6}" y="{YB(vv)+3:.0f}" font-size="8.5" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for who, c in (("na", cna), ("an", can)):
        pts = " ".join(f"{XB(i):.1f},{YB(forageB[nz][who][0]):.1f}" for i, nz in enumerate(NOISES_B))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.4"/>')
        for i, nz in enumerate(NOISES_B):
            e.append(f'<circle cx="{XB(i):.1f}" cy="{YB(forageB[nz][who][0]):.1f}" r="2.6" fill="{c}"/>')
    for i, nz in enumerate(NOISES_B):
        e.append(f'<text x="{XB(i):.0f}" y="{oy+ph+16:.0f}" font-size="8.5" fill="#5b6b8c" text-anchor="middle">{nz:.2f}</text>')
    e.append(f'<text x="{oxB+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">self-motion noise &#8594;</text>')
    # legend
    e.append(f'<rect x="{oxB+pw-150}" y="{oy+8}" width="13" height="4" fill="{cna}"/>'
             f'<text x="{oxB+pw-133}" y="{oy+13}" font-size="9" fill="#28324a">no anchoring (drift)</text>')
    e.append(f'<rect x="{oxB+pw-150}" y="{oy+24}" width="13" height="4" fill="{can}"/>'
             f'<text x="{oxB+pw-133}" y="{oy+29}" font-size="9" fill="#28324a">BVC boundary anchoring</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
