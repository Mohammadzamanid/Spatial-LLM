"""
src/eval/code_necessity.py

WHERE THE POPULATION CODE IS NECESSARY — the sharp claim, after the honest finding that an additive
integrator (NoPE+sum Transformer / raw displacement) ties the grid code on pure path integration.

Path integration is necessary but NOT sufficient for a cognitive map. The map additionally needs a
high-capacity, *remappable* population code — exactly what a metric integrator lacks. We test two
things a raw additive code (the integrator's output: a 2-D displacement) cannot do, however well it
integrates:

  A. MEMORY CAPACITY. Bind K locations one-shot (Hebbian w = code(L)); recall each from a noisy probe.
     A high-dimensional, high-frequency population code (grid/place) pattern-separates nearby
     locations; a smooth 2-D metric (or a smooth lift of it) does not -> capacity collapses with K.
     Honest control: random FOURIER features of the displacement (a *periodic* lift) recover capacity
     — i.e. to match grid cells you must build a grid-like periodic code; a smooth MLP lift does not.

  B. MULTI-MAP STORAGE (REMAPPING) — the decisive, information-theoretic necessity. The SAME trajectory
     gives the SAME displacement in every environment, so ANY deterministic function of displacement
     (raw, RFF, MLP-lift, a NoPE-sum hidden) produces IDENTICAL codes across environments and collides
     when several maps are stored together. Grid/place cells REMAP (an env-dependent phase offset /
     field reassignment), so the same location in two rooms has orthogonal codes. We store M maps over
     recurring locations and measure retrieval: with remapping it holds; without (additive, or grid
     with remapping switched OFF) it falls to ~1/M. Remapping — a property of the population code, not
     of any metric integrator — is what makes multiple maps coexist.

Multi-seed, mean +/- 95% CI. Writes results/code_necessity.json + results/code_necessity.svg.

    python -m src.eval.code_necessity --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import _HexGridModules


# ------------------------------------------------------------------------------ position -> code maps
def grid_code(cx, pos, shift=0.0):
    phi = cx.gains.view(-1, 1, 1) * (pos + shift).unsqueeze(0)      # remap = a spatial phase shift
    return cx._grid_code(phi)


def place_centers(n_side, cover):
    xs = torch.linspace(-cover, cover, n_side)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)        # (C,2)


def place_code(centers, pos, sigma, shift=0.0):
    d2 = ((pos + shift).unsqueeze(1) - centers.unsqueeze(0)) ** 2
    return torch.exp(-d2.sum(-1) / (2 * sigma ** 2))


def rff(pos, W, b):                                                 # random Fourier (periodic) lift of 2-D
    return torch.cos(pos @ W + b) * math.sqrt(2.0 / W.shape[1])


def mlp_lift(pos, net):                                             # smooth (non-periodic) lift of 2-D
    return net(pos)


def nrm(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


# --------------------------------------------------------------- A. one-shot associative memory capacity
def capacity_seed(seed, Ks, R=3.0, dim=384, noise=0.1):
    g = torch.Generator().manual_seed(seed)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    centers = place_centers(20, R)
    W = torch.randn(2, dim, generator=g) * 1.2                      # RFF frequencies (~periodic, scale~1)
    b = torch.rand(1, dim, generator=g) * 2 * math.pi
    mlp = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, dim))  # random smooth lift
    coders = {
        "grid (population)": lambda p: grid_code(cx, p),
        "place (population)": lambda p: place_code(centers, p, 2 * R / 19),
        "additive + RFF lift": lambda p: rff(p, W, b),
        "additive + smooth MLP lift": lambda p: mlp_lift(p, mlp),
        "additive (raw 2-D)": lambda p: p,
    }
    out = {name: {} for name in coders}
    for K in Ks:
        locs = (torch.rand(K, 2, generator=g) * 2 - 1) * R
        for name, code in coders.items():
            with torch.no_grad():
                mem = nrm(code(locs))                               # one-shot Hebbian store (K, dim)
                acc = 0.0
                for _ in range(8):
                    probe = nrm(code(locs + noise * torch.randn(K, 2, generator=g)))
                    acc += (((probe @ mem.t()).argmax(1)) == torch.arange(K)).float().mean().item()
                out[name][K] = acc / 8
    return out


# ------------------------------------------------------- B. multi-map storage (remapping is necessary)
def remap_seed(seed, Ms, K=40, R=3.0, dim=384, noise=0.1):
    g = torch.Generator().manual_seed(1000 + seed)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    centers = place_centers(20, R); C = centers.shape[0]
    locs = (torch.rand(K, 2, generator=g) * 2 - 1) * R              # the SAME K locations recur in every map
    Mmax = max(Ms)
    grid_shift = (torch.rand(Mmax, 2, generator=g) * 2 - 1) * 6.0   # grid realignment (phase offset) per env
    place_perm = torch.stack([torch.randperm(C, generator=g) for _ in range(Mmax)])  # global remap per env
    sigma = 2 * R / 19

    coders = {                                                     # (pos, env index) -> code
        "grid + remap": lambda p, e: grid_code(cx, p, grid_shift[e]),
        "place + remap": lambda p, e: place_code(centers, p, sigma)[:, place_perm[e]],
        "grid, NO remap": lambda p, e: grid_code(cx, p, 0.0),     # remapping switched OFF (ablation)
        "additive (raw 2-D)": lambda p, e: p,                     # cannot remap (same displacement, any env)
    }
    out = {name: {} for name in coders}
    for M in Ms:
        for name, code in coders.items():
            with torch.no_grad():
                mem, ids = [], []
                for e in range(M):                                 # store M maps of K items together
                    mem.append(nrm(code(locs, e))); ids.append(torch.arange(K) + e * K)
                mem = torch.cat(mem); ids = torch.cat(ids)          # (M*K, dim)
                acc = 0.0
                for _ in range(8):
                    pr, tgt = [], []
                    for e in range(M):
                        pr.append(nrm(code(locs + noise * torch.randn(K, 2, generator=g), e)))
                        tgt.append(torch.arange(K) + e * K)
                    pr = torch.cat(pr); tgt = torch.cat(tgt)
                    acc += ((pr @ mem.t()).argmax(1) == tgt).float().mean().item()
                out[name][M] = acc / 8
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def aggregate(per_seed, xs):
    names = list(per_seed[0].keys())
    return {nm: {x: dict(zip(("mean", "ci95"), ci95([s[nm][x] for s in per_seed]))) for x in xs} for nm in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    Ks = [5, 10, 25, 50, 100, 200]
    Ms = [1, 2, 4, 8, 16]
    print(f"CODE NECESSITY — where the population code beats an additive integrator "
          f"(n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 76, flush=True)

    capA = aggregate([capacity_seed(s, Ks) for s in seeds], Ks)
    print("\n[A] one-shot memory capacity — recall accuracy vs # stored locations K:", flush=True)
    print("    " + "code".ljust(28) + "".join(f"K={k}".rjust(11) for k in Ks), flush=True)
    for nm in capA:
        print("    " + nm.ljust(28) + "".join(f"{capA[nm][k]['mean']:.0%}".rjust(11) for k in Ks), flush=True)

    capB = aggregate([remap_seed(s, Ms) for s in seeds], Ms)
    print("\n[B] multi-map storage — retrieval accuracy vs # maps M (K=40 recurring locations):", flush=True)
    print("    " + "code".ljust(28) + "".join(f"M={m}".rjust(11) for m in Ms), flush=True)
    for nm in capB:
        print("    " + nm.ljust(28) + "".join(f"{capB[nm][m]['mean']:.0%}".rjust(11) for m in Ms), flush=True)

    out = {"n_seeds": a.seeds, "Ks": Ks, "Ms": Ms,
           "capacity": capA, "multimap": capB}
    os.makedirs("results", exist_ok=True)
    with open("results/code_necessity.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_necessity(capA, Ks, capB, Ms, "results/code_necessity.svg")
    print("\nwrote results/code_necessity.json and results/code_necessity.svg", flush=True)


PALETTE = ["#e6550d", "#756bb1", "#21908c", "#9aa5b8", "#3b528b", "#c9341a"]


def _panel(e, agg, xs, ox, oy, pw, ph, title, xlabel, logx):
    names = list(agg.keys())
    def X(x):
        if logx:
            lo, hi = math.log(xs[0]), math.log(xs[-1]); return ox + (math.log(x) - lo) / (hi - lo) * pw
        return ox + (xs.index(x)) / (len(xs) - 1) * pw
    def Y(v): return oy + ph - v * ph
    e.append(f'<text x="{ox}" y="{oy-9}" font-size="12" font-weight="700" fill="#0b1324">{title}</text>')
    e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<line x1="{ox}" y1="{Y(vv):.1f}" x2="{ox+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{ox-6}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for x in xs:
        e.append(f'<text x="{X(x):.1f}" y="{oy+ph+13:.1f}" font-size="8.5" fill="#5b6b8c" text-anchor="middle">{x}</text>')
    e.append(f'<text x="{ox+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="9.5" fill="#28324a" text-anchor="middle">{xlabel}</text>')
    for i, nm in enumerate(names):
        col = PALETTE[i % len(PALETTE)]
        band_t = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']+agg[nm][x]['ci95']):.1f}" for x in xs)
        band_b = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']-agg[nm][x]['ci95']):.1f}" for x in reversed(xs))
        e.append(f'<polygon points="{band_t} {band_b}" fill="{col}" opacity="0.12"/>')
        pts = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']):.1f}" for x in xs)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for x in xs:
            e.append(f'<circle cx="{X(x):.1f}" cy="{Y(agg[nm][x]["mean"]):.1f}" r="2.6" fill="{col}"/>')
    ly = oy + 4
    for i, nm in enumerate(names):
        e.append(f'<rect x="{ox+pw-150}" y="{ly}" width="12" height="4" fill="{PALETTE[i % len(PALETTE)]}"/>')
        e.append(f'<text x="{ox+pw-135}" y="{ly+5}" font-size="8.5" fill="#28324a">{nm}</text>'); ly += 12


def svg_necessity(capA, Ks, capB, Ms, out):
    pad = 52; pw = 320; ph = 210; gap = 92
    W = pad + pw + gap + pw + pad; H = 64 + ph + 60
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Where the population code is NECESSARY (an additive integrator cannot follow)</text>')
    e.append('<text x="26" y="43" font-size="10.5" fill="#5b6b8c">a code that ties the grid code on path '
             'integration still fails at memory capacity and multi-map storage &#183; mean &#177; 95% CI</text>')
    _panel(e, capA, Ks, pad, 64, pw, ph, "A &#183; one-shot memory capacity", "# stored locations K", True)
    _panel(e, capB, Ms, pad + pw + gap, 64, pw, ph, "B &#183; multi-map storage (remapping)", "# maps M", True)
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
