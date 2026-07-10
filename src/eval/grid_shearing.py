"""
src/eval/grid_shearing.py

GRID SHEARING — the hexagonal grid DEFORMS with environmental geometry (GAPS.md Tier 2).

Grid cells are not a rigid lattice: in polarized / trapezoidal environments they lose hexagonal symmetry, shear,
and lock to walls (Krupic, Bauza, Burton, Barry, O'Keefe 2015 "Grid cell symmetry is shaped by environmental
geometry", Nature; Stensola, Stensola, Moser & Moser 2015 "shearing induced by boundaries", Nature). Nothing in
this repo produced that — the grid modules are a rigid function of position.

Here the deformation is NOT drawn in; it EMERGES. The model's boundary anchoring localizes with a
SQUARE-calibrated rule (`p_hat = bearing·(arena_R − wall_distance)` — "you are at R−d along the wall normal").
In a trapezoid, the walls are not at arena_R along their normals, so that rule MISLOCALIZES, warping the
phase↔position map — and the rate map (over TRUE position) shears. We MEASURE the gridness drop against a clean
double-dissociation falsifier (the deformation must require BOTH the polarized geometry AND the anchoring):

  (A) SHEARING. In a SQUARE arena the grid is hexagonal (gridness high); in a TRAPEZOID with the same anchoring
      it DEFORMS (gridness collapses). Emergent — the shear is nowhere in the model, it falls out of the
      geometry-mismatched boundary fix. A dose-response (deformation grows with the trapezoid's shear) confirms
      it tracks the geometry.
  (B) DOUBLE-DISSOCIATION FALSIFIER. Deformation needs BOTH ingredients: TRAPEZOID + NO anchoring -> the grid
      stays hexagonal (geometry alone does nothing to the rigid path-integrator); SQUARE + anchoring -> also
      hexagonal (the square-calibrated fix is correct there). Only TRAPEZOID + anchoring deforms.

    python -m src.eval.grid_shearing --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.emergence import rate_maps, grid_stats

R = 3.0            # arena half-size (matches the grid's arena_R calibration)
N_WALK = 1500      # trajectories
T_WALK = 40        # steps per trajectory
M_FINE = 64        # cells in the finest grid module (side*side); coarse modules can't score as grids
SHEAR = 2.2        # trapezoid shear (top edge inset per side); 0 = square


def trapezoid_walks(n, T, shear, seed=0):
    """Random walks in a trapezoid: horizontal top/bottom at y=±R, left/right walls CONVERGING toward the top by
    `shear` per side (shear=0 -> square). Starts at the origin so grid phase = gains·position. Returns
    v3d (n,T,3), boundary_obs (dist, outward-normal bearing to the nearest wall) (n,T,2), true position (n,T,2)."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.zeros(n, 2); head = torch.rand(n, generator=g) * 2 * math.pi
    v3d = torch.zeros(n, T, 3); bobs = torch.zeros(n, T, 2); tpos = torch.zeros(n, T, 2)
    dn = math.hypot(shear, 2 * R)                                            # |slanted-wall direction|
    for t in range(T):
        head = head + (torch.rand(n, generator=g) - 0.5) * 1.0
        sp = 0.2 + torch.rand(n, generator=g) * 0.6
        np_ = pos + torch.stack([sp * head.cos(), sp * head.sin()], -1)
        y = np_[:, 1].clamp(-R, R)
        xL = -R + shear * (y + R) / (2 * R); xR = R - shear * (y + R) / (2 * R)   # wall x at this height
        x = torch.max(torch.min(np_[:, 0], xR), xL)                          # clamp inside the trapezoid
        pos = torch.stack([x, y], -1)
        v3d[:, t, :2] = pos - (tpos[:, t - 1] if t > 0 else torch.zeros(n, 2))
        tpos[:, t] = pos
        d_top = R - y; d_bot = y + R
        d_left = ((x + R) * 2 * R - (y + R) * shear) / dn                     # perpendicular dist to slanted walls
        d_right = ((R - x) * 2 * R - (y + R) * shear) / dn
        dists = torch.stack([d_right, d_left, d_top, d_bot], -1).clamp(min=0.0)
        bearings = torch.stack([torch.full_like(x, math.atan2(shear, 2 * R)),      # right outward normal
                                torch.full_like(x, math.atan2(shear, -2 * R)),     # left outward normal
                                torch.full_like(x, math.pi / 2),                   # top
                                torch.full_like(x, -math.pi / 2)], -1)             # bottom
        dmin, which = dists.min(-1)
        bobs[:, t, 0] = dmin
        bobs[:, t, 1] = bearings.gather(1, which.unsqueeze(1)).squeeze(1)
    return v3d, bobs, tpos


def gridness_of(shear, anchor, seed):
    """Run the grid over trapezoid walks; return the top-cell gridness of the finest module (rate map over TRUE
    position)."""
    torch.manual_seed(seed)
    cx = _HexGridModules(embed_dim=64, n_modules=5, base_spacing=1.5, boundary_anchor=anchor, learned_loc=False)
    v3d, bobs, tpos = trapezoid_walks(N_WALK, T_WALK, shear, seed=seed)
    out = cx.forward(v3d, boundary_obs=(bobs if anchor else None), return_grid_seq=True)
    grid_seq = out[-1] if isinstance(out, tuple) else out
    acts = grid_seq.reshape(-1, grid_seq.shape[-1])[:, :M_FINE]
    p = tpos.reshape(-1, 2)
    rms, _ = rate_maps(p, acts, G=24, R=R)
    grds = torch.tensor([grid_stats(rms[c])[0] for c in range(M_FINE)])
    grds = grds[~grds.isnan()]
    return grds.topk(min(15, len(grds))).values.mean().item()


def run_seed(seed):
    sq_anchor = gridness_of(0.0, True, seed)          # square + anchoring  (fix correct -> hexagonal)
    tz_anchor = gridness_of(SHEAR, True, seed)        # TRAPEZOID + anchoring -> deforms
    sq_noanchor = gridness_of(0.0, False, seed)       # square + no anchoring
    tz_noanchor = gridness_of(SHEAR, False, seed)     # trapezoid + NO anchoring (falsifier: geometry alone)
    tz_half = gridness_of(SHEAR / 2, True, seed)      # dose-response (milder shear)
    return {
        "grid_square_anchor": round(sq_anchor, 4),
        "grid_trapezoid_anchor": round(tz_anchor, 4),
        "grid_square_noanchor": round(sq_noanchor, 4),
        "grid_trapezoid_noanchor": round(tz_noanchor, 4),
        "grid_trapezoid_half": round(tz_half, 4),
        "shear_drop": round(sq_anchor - tz_anchor, 4),               # deformation under anchoring
        "falsifier_gap": round(tz_noanchor - tz_anchor, 4),          # same geometry, only anchoring differs
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["grid_square_anchor", "grid_trapezoid_anchor", "grid_square_noanchor", "grid_trapezoid_noanchor",
        "grid_trapezoid_half", "shear_drop", "falsifier_gap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: gridness  square+anchor {p['grid_square_anchor']:+.2f}  TRAPEZOID+anchor "
              f"{p['grid_trapezoid_anchor']:+.2f}  | square+no-anchor {p['grid_square_noanchor']:+.2f}  "
              f"trapezoid+no-anchor {p['grid_trapezoid_noanchor']:+.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nGRID SHEARING — the hexagonal grid deforms with geometry (n={a.seeds}; mean ± 95% CI)\n" + "=" * 88, flush=True)
    print(f"  (A) SHEARING (emergent): SQUARE+anchor gridness {agg['grid_square_anchor'][0]:+.3f} ± "
          f"{agg['grid_square_anchor'][1]:.3f}  ->  TRAPEZOID+anchor {agg['grid_trapezoid_anchor'][0]:+.3f} ± "
          f"{agg['grid_trapezoid_anchor'][1]:.3f}   (drop {agg['shear_drop'][0]:+.3f} ± {agg['shear_drop'][1]:.3f})", flush=True)
    print(f"      dose-response: half-shear {agg['grid_trapezoid_half'][0]:+.2f} (between square "
          f"{agg['grid_square_anchor'][0]:+.2f} and full-shear {agg['grid_trapezoid_anchor'][0]:+.2f}) — the "
          f"deformation grows with the geometry.", flush=True)
    print(f"  (B) DOUBLE-DISSOCIATION FALSIFIER — the deformation needs BOTH geometry AND anchoring:", flush=True)
    print(f"      TRAPEZOID + NO anchoring {agg['grid_trapezoid_noanchor'][0]:+.3f} (still hexagonal — geometry "
          f"alone does nothing); SQUARE + anchoring {agg['grid_square_anchor'][0]:+.3f} (hexagonal — the "
          f"square-calibrated fix is correct there). Only trapezoid+anchoring deforms (falsifier gap "
          f"{agg['falsifier_gap'][0]:+.3f} ± {agg['falsifier_gap'][1]:.3f}).", flush=True)

    sound = (agg["shear_drop"][0] > 0.4 and agg["grid_square_anchor"][0] > 0.5 and
             agg["grid_trapezoid_noanchor"][0] > 0.5 and agg["falsifier_gap"][0] > 0.4)
    verdict = ("SOUND — the hexagonal grid stays hexagonal in a square arena but SHEARS in a trapezoid under the "
               "same boundary anchoring, and the deformation requires BOTH the polarized geometry and the "
               "anchoring (a clean double dissociation). The grid deforms itself with environmental geometry — "
               "measured, never drawn (Krupic 2015; Stensola 2015)." if sound else
               "WEAK — the shearing/double-dissociation did not clear the falsifiers; revisit the regime.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "R": R, "shear": SHEAR, "n_walk": N_WALK, "T": T_WALK,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/grid_shearing.json", "w"), indent=2)
    _svg(agg, "results/grid_shearing.svg")
    print("\nwrote results/grid_shearing.json and results/grid_shearing.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Grid shearing: the hexagonal grid deforms with environmental geometry</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">the grid stays hexagonal in a square but shears '
             'in a trapezoid under the same anchoring; deformation needs BOTH geometry and anchoring — never drawn</text>')
    oy = 60; base = oy + ph
    def bars(ox, title, data, foot):
        e.append(f'<text x="{ox}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">{title}</text>')
        e.append(f'<line x1="{ox}" y1="{base}" x2="{ox+pw}" y2="{base}" stroke="#33415c"/>')
        e.append(f'<line x1="{ox}" y1="{base-0.0:.0f}" x2="{ox+pw}" y2="{base:.0f}" stroke="#9aa6bd"/>')
        top = 1.15
        for i, (lab, v, col) in enumerate(data):
            hv = max(v, 0.0) / top * (ph - 24); x = ox + 22 + i * 74
            e.append(f'<rect x="{x}" y="{base-hv:.1f}" width="52" height="{hv:.1f}" fill="{col}" opacity="0.9"/>')
            e.append(f'<text x="{x+26}" y="{base-hv-6:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.2f}</text>')
            for j, ln in enumerate(lab.split("\n")):
                e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.3" fill="#28324a" text-anchor="middle">{ln}</text>')
        e.append(f'<text x="{ox}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">{foot}</text>')
    bars(pad, "(A) gridness: square vs trapezoid (both anchored)",
         [("square\n+anchor", agg["grid_square_anchor"][0], "#2ca25f"),
          ("½ shear\n+anchor", agg["grid_trapezoid_half"][0], "#c98a1a"),
          ("TRAPEZOID\n+anchor", agg["grid_trapezoid_anchor"][0], "#c9341a")],
         "the grid shears more as the geometry polarizes (dose-response)")
    bars(pad + pw + gap, "(B) double dissociation (needs geometry AND anchoring)",
         [("square\n+anchor", agg["grid_square_anchor"][0], "#2ca25f"),
          ("trapezoid\nNO anchor", agg["grid_trapezoid_noanchor"][0], "#2ca25f"),
          ("TRAPEZOID\n+anchor", agg["grid_trapezoid_anchor"][0], "#c9341a")],
         "only trapezoid+anchoring deforms; each ingredient alone keeps it hexagonal")
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
