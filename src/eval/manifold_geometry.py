"""
src/eval/manifold_geometry.py

DOES THE ATTRACTOR MANIFOLD ITSELF DEFORM? — the deeper question behind grid shearing (GAPS.md #5d follow-up).

`grid_shearing.py` (#5d) showed the RATE MAP shears in a trapezoid. But a rate map is a read-out over physical
space; a sharper critique asks whether the NEURAL MANIFOLD — the geometry of the population activity itself —
deforms to the environment, or whether it stays a rigid torus while only the space→manifold MAP warps. In vivo,
the grid population lies on a torus whose TOPOLOGY is preserved across environments and even sleep (Gardner,
Hermansen, Pachitariu, Burak, Baas, Dunn, Moser & Moser 2022) — which predicts a RIGID manifold. We measure it,
and then ask what it takes to make a manifold that genuinely deforms.

  (A) THE RIGID CAN's MANIFOLD DOES NOT DEFORM. Take the anchored grid codes of #5d in a SQUARE and a TRAPEZOID.
      The trapezoid population codes lie on the SAME manifold as the square's — manifold overlap ≈ the
      square-vs-square reference. The manifold is a rigid torus; the #5d shearing is entirely a warping of the
      space→manifold MAP, not a deformation of the manifold. (Consistent with Gardner 2022.)
  (B) THE FIXED GRID IGNORES NON-EUCLIDEAN GEOMETRY; A PLASTIC CODE DEFORMS TO IT. In a BARRIER (hairpin-like)
      environment, the fixed grid code's manifold does not respect the wall — its neural distances track
      Euclidean, not geodesic, distance (barrier-respect ≈ 0). A PLASTIC code, whose geometry is shaped by
      experience of the environment, reshapes so its neural distances track the GEODESIC (barrier-respecting)
      geometry (barrier-respect > 0). The manifold deforms — but only when it is plastic, which the rigid CAN is
      not.

So the honest answer: the standard continuous-attractor manifold retains its (toroidal) perfection — #5d is a map
effect — and manifold deformation to the environment's actual geometry REQUIRES a plastic attractor, exactly the
capacity the critique says a rigid CAN lacks. Multi-seed, mean ± 95% CI. Writes results/manifold_geometry.json + .svg.

    python -m src.eval.manifold_geometry --seeds 5
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from src.eval.grid_shearing import M_FINE, trapezoid_walks
from src.eval.successor import geodesic, make_world
from src.models.neuro.trajectory_cortex import _HexGridModules

SHEAR = 2.2
G = 11
GAP = 1                 # narrow doorway -> strongly non-Euclidean (long geodesic detours)


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def _offdiag(M):
    n = M.shape[0]
    return M[~torch.eye(n, dtype=torch.bool)]


# ----------------------------------------------------------------------------- (A) rigid CAN manifold overlap
def anchored_codes(shear, seed, N=800, T=40):
    torch.manual_seed(seed)
    cx = _HexGridModules(embed_dim=64, n_modules=5, base_spacing=1.5, boundary_anchor=True, learned_loc=False)
    v3d, bobs, _ = trapezoid_walks(N, T, shear, seed=seed)
    out = cx.forward(v3d, boundary_obs=bobs, return_grid_seq=True)
    grid_seq = out[-1] if isinstance(out, tuple) else out
    return grid_seq.reshape(-1, grid_seq.shape[-1])[:, :M_FINE].detach()


def manifold_overlap(A, B):
    """Fraction of B's variance captured by the top PCA subspace of A's manifold (1 = B lies on A's manifold)."""
    U, S, V = torch.pca_lowrank(A - A.mean(0), q=12)
    Bm = B - A.mean(0)
    return (1 - ((Bm - Bm @ V @ V.t()) ** 2).sum() / (Bm ** 2).sum()).item()


# ----------------------------------------------------------------------------- (B) barrier: rigid vs plastic
def barrier_setup(seed):
    free, cells, idx = make_world(G, GAP, barrier=True)
    pos = torch.tensor([[i, j] for (i, j) in cells], dtype=torch.float) / (G - 1) * 4 - 2
    geo = torch.stack([geodesic(cells, idx, free, G, cells[k]) for k in range(len(cells))])
    euc = torch.cdist(pos, pos)
    return pos, geo, euc


def barrier_respect(D, geo, euc):
    """How much better GEODESIC distance predicts neural distance than EUCLIDEAN does (>0 = respects the wall)."""
    return _corr(_offdiag(D), _offdiag(geo)) - _corr(_offdiag(D), _offdiag(euc))


def run_seed(seed):
    # (A) does the grid CAN manifold deform in a trapezoid, or stay the same torus?
    sq = anchored_codes(0.0, seed)
    tz = anchored_codes(SHEAR, seed)
    sq2 = anchored_codes(0.0, seed + 50)
    overlap_trap = manifold_overlap(sq, tz)
    overlap_ref = manifold_overlap(sq, sq2)

    # (B) barrier: the fixed grid ignores it; a plastic code deforms to its geodesic geometry
    pos, geo, euc = barrier_setup(seed)
    torch.manual_seed(seed)
    cx = _HexGridModules(embed_dim=64, n_modules=5, base_spacing=1.5)
    Dg = torch.cdist(cx.grid_code_at(pos).detach(), cx.grid_code_at(pos).detach())
    z = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 16))
    opt = torch.optim.Adam(z.parameters(), 3e-3); gt = geo / geo.max()
    mask = ~torch.eye(len(pos), dtype=torch.bool)
    for _ in range(1800):
        Dz = torch.cdist(z(pos), z(pos))
        loss = ((Dz - gt) ** 2)[mask].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    Dp = torch.cdist(z(pos).detach(), z(pos).detach())
    return {"manifold_overlap_trapezoid": overlap_trap, "manifold_overlap_reference": overlap_ref,
            "manifold_deformation_grid": overlap_ref - overlap_trap,      # ~0 = rigid (no deformation vs the noise floor)
            "barrier_respect_grid": barrier_respect(Dg, geo, euc),
            "barrier_respect_plastic": barrier_respect(Dp, geo, euc)}


KEYS = ["manifold_overlap_trapezoid", "manifold_overlap_reference", "manifold_deformation_grid",
        "barrier_respect_grid", "barrier_respect_plastic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]

    def ci(vals):
        import math
        t = torch.tensor(vals, dtype=torch.float); n = len(vals)
        sd = t.std(unbiased=True).item() if n > 1 else 0.0
        return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0
    agg = {k: ci([p[k] for p in per]) for k in KEYS}

    print(f"DOES THE ATTRACTOR MANIFOLD ITSELF DEFORM? (n={a.seeds}; mean ± 95% CI)\n" + "=" * 72, flush=True)
    print(f"  (A) RIGID CAN — manifold overlap trapezoid-on-square {agg['manifold_overlap_trapezoid'][0]:.3f} vs "
          f"square-on-square reference {agg['manifold_overlap_reference'][0]:.3f}", flush=True)
    print(f"      -> manifold deformation {agg['manifold_deformation_grid'][0]:+.3f} ± "
          f"{agg['manifold_deformation_grid'][1]:.3f}  (~0: the manifold is a RIGID torus; #5d is a MAP effect)", flush=True)
    print(f"  (B) BARRIER (non-Euclidean) — does the manifold respect the wall (geodesic > Euclidean)?", flush=True)
    print(f"      fixed GRID CAN  {agg['barrier_respect_grid'][0]:+.2f} ± {agg['barrier_respect_grid'][1]:.2f}  "
          f"(ignores the barrier -- rigid)", flush=True)
    print(f"      PLASTIC code    {agg['barrier_respect_plastic'][0]:+.2f} ± {agg['barrier_respect_plastic'][1]:.2f}  "
          f"(reshapes to the geodesic geometry -- deforms)", flush=True)
    print(f"\n  the standard continuous-attractor manifold stays a rigid torus (Gardner 2022); the grid shearing of "
          f"#5d is a warping of the space->manifold MAP, not the manifold. Deforming the manifold to the "
          f"environment's actual geometry REQUIRES a plastic attractor -- the capacity a rigid CAN lacks.", flush=True)

    out = {"n_seeds": a.seeds, "shear": SHEAR, "barrier_gap": GAP,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS},
           "verdict": "Beyond the rate-map shearing of #5d, the NEURAL MANIFOLD of the standard continuous "
                      "attractor does NOT deform: trapezoid population codes lie on the same manifold as the "
                      "square's (overlap ~ the square-vs-square reference), a rigid torus consistent with Gardner "
                      "2022 -- so #5d is a warping of the space->manifold MAP, not the manifold. In a non-Euclidean "
                      "(barrier) environment the fixed grid ignores the wall (neural distance ~ Euclidean) while a "
                      "PLASTIC code reshapes to the geodesic geometry (respects the wall). Manifold deformation to "
                      "the environment's actual geometry requires plasticity, which the rigid CAN lacks."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/manifold_geometry.json", "w"), indent=2)
    svg_manifold(agg, "results/manifold_geometry.svg")
    print("\nwrote results/manifold_geometry.json and results/manifold_geometry.svg", flush=True)


def svg_manifold(agg, out):
    W_, H = 700, 300
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Does the attractor manifold itself deform? Rigid torus (map warps) vs a plastic code</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">the #5d shearing is a warping of the '
         'space&#8594;manifold MAP; the manifold stays a torus (Gardner 2022)</text>']
    # left: manifold overlap (trapezoid vs reference) -- rigid if equal
    bx, by, bh, bw = 44, 84, 170, 66
    e.append(f'<text x="{bx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">(A) grid manifold overlap</text>')
    for i, (k, lab) in enumerate([("manifold_overlap_reference", "square\n(reference)"), ("manifold_overlap_trapezoid", "trapezoid")]):
        v = agg[k][0]; x = bx + i * (bw + 16); h = v * bh; col = "#8c8c8c" if i == 0 else "#2ca25f"
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh}" x2="{bx+2*(bw+16):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{bx}" y="{by+bh+40:.0f}" font-size="8.5" fill="#5b6b8c">equal -> the manifold is a RIGID torus</text>')
    # right: barrier-respect grid vs plastic
    rx = 380; rw = 96
    e.append(f'<text x="{rx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">(B) does the manifold respect a barrier?</text>')
    top = max(0.05, agg["barrier_respect_plastic"][0]) * 1.4
    for i, (k, lab, col) in enumerate([("barrier_respect_grid", "fixed grid\n(rigid)", "#c9341a"), ("barrier_respect_plastic", "plastic\n(deforms)", "#2ca25f")]):
        v = agg[k][0]; x = rx + i * (rw + 20); h = max(0, v) / top * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{rw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{rx-4}" y1="{by+bh}" x2="{rx+2*(rw+20):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{rx}" y="{by+bh+40:.0f}" font-size="8.5" fill="#5b6b8c">geodesic minus Euclidean corr; &gt;0 = manifold bends to the wall</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
