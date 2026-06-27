"""
src/eval/grid_capacity.py

GRID-CELL CODING CAPACITY (Fiete) — why the brain uses a periodic multi-scale code for space.

At a FIXED neuron budget, a periodic multi-scale GRID code resolves a large arena combinatorially
(each module's phase is reused across the whole space; modules combine like a residue number system),
whereas a local PLACE code (same budget) must tile the arena with bumps and therefore resolves it ever
more coarsely as the arena grows. The classic prediction (Sreenivasan & Fiete 2011; Fiete et al. 2008):
grid capacity scales EXPONENTIALLY with neuron count, place only LINEARLY — so as the arena scales at a
fixed budget, grid local resolution stays ~constant while place resolution degrades ~linearly.

We measure capacity two ways, and they tell a two-part story:

  (A) FISHER INFORMATION (decoder-agnostic). The Cramer-Rao bound: the position precision AVAILABLE in
      the code, independent of any decoder. res = det(Fisher)^(-1/4) (geometric-mean SD of the 2D
      error ellipse; lower = finer). For the cos/sin grid code Fisher = F^T F is position-INDEPENDENT and
      set by the finest period -> flat in L. For the place code Fisher ~ 1/sigma^2 with sigma ∝ L ->
      res ∝ L. PREDICTION: grid flat, place linear; the grid advantage GROWS with arena size.

  (B) LINEAR DECODE (the honest caveat). A simple linear reader CANNOT access the grid code's
      combinatorial capacity (the phase->position map is nonlinear/periodic): linear-decode MAE is
      actually WORSE for grid than place. The information is in the code (A) but is locked behind a
      nonlinear/Bayesian decoder. This is itself the point: grid capacity is real but not free.

Both Fisher forms are verified against autograd in tests. Multi-seed, mean +/- 95% CI.
Writes results/grid_capacity.json + .svg.

    python -m src.eval.grid_capacity --seeds 5
"""
import argparse
import json
import math
import os

import torch

BUDGET = 128                      # matched neuron budget for grid and place
ARENAS = [1.0, 2.0, 4.0, 8.0]     # arena half-width L (arena = [-L, L]^2); width 2L grows 8x
LAMBDA_MIN = 0.5                  # finest grid period (fixed -> sets the resolution floor)


# ---- grid code: periodic multi-scale (cos/sin of phase across K modules) ----
def grid_F(L, gen):
    """K module wave-vectors F_k = (2*pi/period_k) * unit(angle_k); periods log-spaced LAMBDA_MIN..3L."""
    K = BUDGET // 2
    periods = torch.exp(torch.linspace(math.log(LAMBDA_MIN), math.log(3.0 * L), K))
    ang = torch.rand(K, generator=gen) * math.pi
    return torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / periods).unsqueeze(1)  # (K,2)


def grid_code(P, F):
    proj = P @ F.t()
    return torch.cat([proj.cos(), proj.sin()], -1)                 # (N, 2K)


def grid_fisher(F):
    """Fisher = sum_k F_k F_k^T (position-independent: sin^2+cos^2=1). Verified vs autograd in tests."""
    return F.t() @ F                                               # (2,2)


# ---- place code: local Gaussian bumps tiling the arena ----
def place_centers(L):
    n = int(BUDGET ** 0.5); xs = torch.linspace(-L, L, n)
    cx, cy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([cx.reshape(-1), cy.reshape(-1)], -1), (2 * L / (n - 1)) * 1.1   # centers, sigma


def place_code(P, C, sig):
    return torch.exp(-((P.unsqueeze(1) - C.unsqueeze(0)) ** 2).sum(-1) / (2 * sig ** 2))   # (N, M)


def place_fisher(x, C, sig):
    """Fisher(x) = sum_i (r_i^2/sig^4) (x-c_i)(x-c_i)^T. Verified vs autograd in tests."""
    d = x - C
    r = torch.exp(-(d ** 2).sum(-1) / (2 * sig ** 2))
    g = (r / sig ** 2).unsqueeze(-1) * d                          # grad of r_i wrt x  (sign squares away)
    return g.t() @ g                                              # (2,2)


def det_res(J):
    """Cramer-Rao local resolution: geometric-mean SD of the 2D error ellipse (lower = finer)."""
    return torch.det(J).clamp_min(1e-30).item() ** -0.25


# ---- linear decode (the honest caveat: a linear reader can't use grid capacity) ----
def decode_mae(code, P):
    n = P.shape[0]; tr = n // 2
    Ab = torch.cat([code, torch.ones(n, 1)], 1)
    W = torch.linalg.lstsq(Ab[:tr], P[:tr]).solution
    return (Ab[tr:] @ W - P[tr:]).abs().mean().item()


def run_seed(seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in ARENAS:
        X = (torch.rand(400, 2, generator=g) * 2 - 1) * L          # positions for Fisher (median over space)
        P = (torch.rand(4000, 2, generator=g) * 2 - 1) * L         # positions for the linear decode
        F = grid_F(L, g)
        C, sig = place_centers(L)
        gJ = grid_fisher(F)
        g_res = torch.tensor([det_res(gJ)]).median().item()        # position-independent
        p_res = torch.tensor([det_res(place_fisher(x, C, sig)) for x in X]).median().item()
        g_mae = decode_mae(grid_code(P, F), P)
        p_mae = decode_mae(place_code(P, C, sig), P)
        out[L] = {"grid_res": g_res, "place_res": p_res, "grid_mae": g_mae, "place_mae": p_mae}
    return out


def loglog_slope(ys):
    """least-squares slope of log(y) vs log(L) across ARENAS (0 = flat, 1 = linear in arena size)."""
    xs = torch.tensor([math.log(L) for L in ARENAS]); ly = torch.tensor([math.log(y) for y in ys])
    xs = xs - xs.mean(); return (xs @ (ly - ly.mean()) / (xs @ xs)).item()


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 4), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 4) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    runs = [run_seed(s) for s in range(a.seeds)]
    keys = ["grid_res", "place_res", "grid_mae", "place_mae"]
    perL = {L: {k: ci([r[L][k] for r in runs]) for k in keys} for L in ARENAS}
    slopes = {k: ci([loglog_slope([r[L][k] for L in ARENAS]) for r in runs]) for k in keys}

    print(f"\nGRID-CELL CODING CAPACITY (Fiete) — fixed {BUDGET}-unit budget, n={a.seeds} (mean ± 95% CI)\n" + "=" * 78, flush=True)
    print("(A) FISHER-INFO local resolution (decoder-agnostic; lower = finer):", flush=True)
    for L in ARENAS:
        gr, pr = perL[L]["grid_res"], perL[L]["place_res"]
        print(f"    arena width {2*L:>2.0f}: grid {gr[0]:.4f}±{gr[1]:.4f} | place {pr[0]:.4f}±{pr[1]:.4f}"
              f" | place/grid {pr[0]/gr[0]:5.1f}x", flush=True)
    print(f"    scaling (log-log slope vs arena): grid {slopes['grid_res'][0]:+.2f}±{slopes['grid_res'][1]:.2f}"
          f"  place {slopes['place_res'][0]:+.2f}±{slopes['place_res'][1]:.2f}   (0=flat, 1=linear)", flush=True)
    print("\n(B) LINEAR-DECODE MAE (the honest caveat; a linear reader can't use grid capacity):", flush=True)
    for L in ARENAS:
        gm, pm = perL[L]["grid_mae"], perL[L]["place_mae"]
        print(f"    arena width {2*L:>2.0f}: grid {gm[0]:.3f}±{gm[1]:.3f} | place {pm[0]:.3f}±{pm[1]:.3f}", flush=True)
    rfac = perL[8.0]["place_res"][0] / perL[8.0]["grid_res"][0]
    print(f"\n  -> (A) the grid code holds local resolution ~CONSTANT as the arena grows "
          f"(slope {slopes['grid_res'][0]:+.2f}) while place degrades ~LINEARLY (slope {slopes['place_res'][0]:+.2f}); "
          f"the grid advantage GROWS to {rfac:.0f}x at the largest arena -- exponential vs linear capacity (Fiete). "
          f"(B) BUT a linear reader can't extract it (grid-MAE > place-MAE): the capacity is real, not free -- "
          f"it requires a nonlinear/Bayesian decoder.", flush=True)

    out = {"n_seeds": a.seeds, "budget": BUDGET, "arenas": ARENAS,
           "per_arena": {str(L): perL[L] for L in ARENAS}, "loglog_slope": slopes}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/grid_capacity.json", "w"), indent=2)
    svg(perL, slopes, "results/grid_capacity.svg")
    print("\nwrote results/grid_capacity.json and results/grid_capacity.svg", flush=True)


def svg(perL, slopes, out):
    import math as _m
    pad = 58; pw = 300; ph = 210; gap = 96; W = pad + 2 * pw + gap + 30; H = 78 + ph + 46
    col = {"grid": "#2ca25f", "place": "#c9341a"}
    ys_all = [perL[L][k][0] for L in ARENAS for k in ("grid_res", "place_res", "grid_mae", "place_mae")]
    lo = min(ys_all) * 0.7; hi = max(ys_all) * 1.4
    def Yl(v, oy): return oy + ph - (_m.log(max(v, 1e-9)) - _m.log(lo)) / (_m.log(hi) - _m.log(lo)) * ph
    def Xl(i): return (i / (len(ARENAS) - 1)) * pw
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Grid-cell coding capacity: periodic multi-scale code resolves large space at fixed budget (Fiete)</text>')
    e.append(f'<text x="26" y="44" font-size="10.5" fill="#5b6b8c">{BUDGET}-unit budget, matched. '
             'log axes; lower = better. Arena width grows 8&#215;.</text>')
    panels = [("grid_res", "place_res", "(A) Fisher-info resolution (decoder-agnostic) &#8212; capacity",
               f"grid slope {slopes['grid_res'][0]:+.2f} (flat) vs place {slopes['place_res'][0]:+.2f} (linear)"),
              ("grid_mae", "place_mae", "(B) Linear-decode MAE &#8212; the honest caveat",
               "a linear reader can&#8217;t use grid capacity (grid &#62; place)")]
    for pi, (gk, pk, title, sub) in enumerate(panels):
        ox = pad + pi * (pw + gap); oy = 64
        e.append(f'<text x="{ox}" y="{oy-6}" font-size="11.5" font-weight="700" fill="#0b1324">{title}</text>')
        e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
                 f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
        for dec in (0.01, 0.1, 1.0):
            if lo <= dec <= hi:
                yy = Yl(dec, oy)
                e.append(f'<line x1="{ox}" y1="{yy:.0f}" x2="{ox+pw}" y2="{yy:.0f}" stroke="#eef1f6"/>'
                         f'<text x="{ox-6}" y="{yy+3:.0f}" font-size="8.5" fill="#5b6b8c" text-anchor="end">{dec:g}</text>')
        for i, L in enumerate(ARENAS):
            e.append(f'<text x="{ox+Xl(i):.0f}" y="{oy+ph+14:.0f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{2*L:.0f}</text>')
        e.append(f'<text x="{ox+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">arena width &#8594;</text>')
        for who, kk in (("grid", gk), ("place", pk)):
            pts = " ".join(f"{ox+Xl(i):.1f},{Yl(perL[L][kk][0], oy):.1f}" for i, L in enumerate(ARENAS))
            e.append(f'<polyline points="{pts}" fill="none" stroke="{col[who]}" stroke-width="2.6"/>')
            for i, L in enumerate(ARENAS):
                e.append(f'<circle cx="{ox+Xl(i):.1f}" cy="{Yl(perL[L][kk][0], oy):.1f}" r="2.6" fill="{col[who]}"/>')
        e.append(f'<text x="{ox+8}" y="{oy+14}" font-size="9" fill="#7787a6">{sub}</text>')
    ly = 64
    for who, lab in (("grid", "grid code (periodic, multi-scale)"), ("place", "place code (local bumps)")):
        e.append(f'<rect x="{pad+2*pw+gap-150}" y="{ly}" width="14" height="4" fill="{col[who]}"/>')
        e.append(f'<text x="{pad+2*pw+gap-132}" y="{ly+5}" font-size="9" fill="#28324a">{lab}</text>'); ly += 16
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
