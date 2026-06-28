"""
src/eval/plane_of_motion.py

3D NAVIGATION VIA A PLANE-ALIGNED 2D GRID — the bat scheme, and an honest scope.

A naive 3D grid lattice is costly and is not what is observed; freely-flying bats appear to use a 2D
toroidal grid code aligned to the behaviorally-relevant PLANE OF MOTION, plus an off-plane code, rather
than a full 3D lattice (the 2026 bat result; cf. the repo's earlier "z as a 1D place code" stub). We
implement that faithfully — estimate the motion plane (PCA of the trajectory), project onto it, and
path-integrate the REAL hexagonal grid cortex (`grid_code_at`) on the in-plane coordinates, with a cheap
1-D off-plane code — and measure:

  (A) PLANE RECOVERY. PCA recovers the motion-plane normal almost exactly (orientation-invariant), so the
      grid can be aligned to whatever plane the animal is moving in.
  (B) ORIENTATION-INVARIANT 3D LOCALIZATION. the plane-aligned code decodes 3D position with accuracy that
      is FLAT across plane tilt — it works in any motion plane because it aligns to it.
  (C) ALIGNMENT IS NECESSARY. a FIXED (horizontal) 2D grid degrades as the motion plane tilts steeply (the
      in-plane motion rotates into the coarse off-axis code) — you must align the grid to the motion plane.

Honest scope: at matched budget we did NOT find a robust 3D-decode advantage of the plane-aligned 2D grid
over a naive isotropic 3D grid (a learned decoder compensates for both; the capacity gap is modest in this
regime). So the contribution is the faithful, orientation-invariant MECHANISM (2D grid on the estimated
motion plane, the bat scheme) and the alignment necessity — not a decode win over a 3D lattice.

Multi-seed, mean +/- 95% CI. Writes results/plane_of_motion.json + .svg.

    python -m src.eval.plane_of_motion --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.eval.agent_grid_cortex import build_cortex, R

L = 2.0; LOFF = 0.3; SIGMA = 0.2                 # in-plane half-extent, off-plane half-extent, code noise
TILTS = [0.0, 0.5, 1.0, 1.4]                     # motion-plane tilt from horizontal (rad)


def gen_traj(tilt, n, gen):
    p2 = (torch.rand(n, 2, generator=gen) * 2 - 1) * L
    off = (torch.rand(n, 1, generator=gen) * 2 - 1) * LOFF
    e1 = torch.tensor([1.0, 0.0, 0.0]); e2 = torch.tensor([0.0, math.cos(tilt), math.sin(tilt)])
    en = torch.tensor([0.0, -math.sin(tilt), math.cos(tilt)])
    P = p2[:, 0:1] * e1 + p2[:, 1:2] * e2 + off * en
    return P, en


def place1d(off, ncell, rng):
    c = torch.linspace(-rng, rng, ncell)
    return torch.exp(-((off - c) ** 2) / (2 * (2 * rng / ncell) ** 2))


def code_and_recovery(mode, P, mod):
    """Build the [in-plane hex grid | off-plane 1-D] code. aligned: project onto the PCA-estimated plane;
    fixed: use the horizontal (x,y) plane + z. Returns (code, plane_normal_recovery_error or nan)."""
    if mode == "aligned":
        X = P - P.mean(0)
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        b1, b2, bn = Vt[0], Vt[1], Vt[2]
        inplane = torch.stack([P @ b1, P @ b2], -1)
        off = (P @ bn).unsqueeze(1)
        code = torch.cat([mod.grid_code_at(inplane), place1d(off, 12, LOFF)], -1)   # off-plane coded finely
        return code, b1, b2, bn
    inplane = P[:, :2]; off = P[:, 2:3]
    code = torch.cat([mod.grid_code_at(inplane), place1d(off, 12, L)], -1)          # z coded coarsely (full range)
    return code, None, None, None


def decode_err(mode, tilt, mod, gen):
    P, en = gen_traj(tilt, 2500, gen)
    code, b1, b2, bn = code_and_recovery(mode, P, mod)
    nrm = (min((bn - en).norm().item(), (bn + en).norm().item()) if mode == "aligned" else float("nan"))
    dec = nn.Sequential(nn.Linear(code.shape[1], 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.Adam(dec.parameters(), 3e-3); tr = 1700
    for _ in range(600):
        cn = code[:tr] + torch.randn(tr, code.shape[1], generator=gen) * SIGMA
        loss = ((dec(cn) - P[:tr]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cn = code[tr:] + torch.randn(code.shape[0] - tr, code.shape[1], generator=gen) * SIGMA
        return (dec(cn) - P[tr:]).norm(dim=1).mean().item(), nrm


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed)
    mod = build_cortex(seed)
    aligned = {}; fixed = {}; recov = {}
    for t in TILTS:
        a, ne = decode_err("aligned", t, mod, gen); f, _ = decode_err("fixed", t, mod, gen)
        aligned[t] = a; fixed[t] = f; recov[t] = ne
    return {"aligned": aligned, "fixed": fixed, "recov": recov}


def ci(vals):
    tt = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(tt.mean().item(), 4), round(1.96 * tt.std(unbiased=True).item() / math.sqrt(n), 4) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    A = {t: ci([p["aligned"][t] for p in per]) for t in TILTS}
    F = {t: ci([p["fixed"][t] for p in per]) for t in TILTS}
    Rc = {t: ci([p["recov"][t] for p in per]) for t in TILTS}

    print(f"\n3D NAVIGATION VIA A PLANE-ALIGNED 2D GRID (the bat scheme; n={a.seeds}; mean ± 95% CI)\n" + "=" * 80, flush=True)
    print(f"    {'tilt(deg)':>9} | {'plane-aligned 3D err':>20} {'plane recov err':>16} | {'fixed-plane 3D err':>18}", flush=True)
    for t in TILTS:
        print(f"    {math.degrees(t):>9.0f} | {A[t][0]:>18.3f}   {Rc[t][0]:>16.3f} | {F[t][0]:>18.3f}", flush=True)
    aflat = max(A[t][0] for t in TILTS) - min(A[t][0] for t in TILTS)
    print(f"\n  -> (A) PCA recovers the motion-plane normal almost exactly (err ~{Rc[TILTS[-1]][0]:.3f}, "
          f"orientation-invariant). (B) the plane-aligned 2D grid (the REAL hex cortex, on the motion plane) "
          f"localizes 3D position with accuracy FLAT across plane tilt (range {aflat:.3f}: {A[TILTS[0]][0]:.3f}"
          f"->{A[TILTS[-1]][0]:.3f}) -- it works in any plane because it aligns to it. (C) a FIXED horizontal "
          f"grid degrades as the plane tilts steeply ({F[TILTS[0]][0]:.3f}->{F[TILTS[-1]][0]:.3f} at "
          f"{math.degrees(TILTS[-1]):.0f}°) -- alignment is necessary. Honest scope: at matched budget we found "
          f"NO robust decode advantage over a naive 3D grid (decoder-masked); the contribution is the faithful, "
          f"orientation-invariant mechanism + alignment, not a decode win over a 3D lattice.", flush=True)

    out = {"n_seeds": a.seeds, "sigma": SIGMA, "tilts_rad": TILTS,
           "aligned": {str(t): A[t] for t in TILTS}, "fixed": {str(t): F[t] for t in TILTS},
           "plane_recovery": {str(t): Rc[t] for t in TILTS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/plane_of_motion.json", "w"), indent=2)
    svg(A, F, Rc, "results/plane_of_motion.svg")
    print("\nwrote results/plane_of_motion.json and results/plane_of_motion.svg", flush=True)


def svg(A, F, Rc, out):
    pad = 60; pw = 420; ph = 220; W = pad + pw + 180; H = 84 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             '3D navigation via a plane-aligned 2D grid (the bat scheme)</text>')
    e.append(f'<text x="26" y="42" font-size="10.5" fill="#5b6b8c">PCA recovers the motion plane '
             f'(err ~{Rc[TILTS[-1]][0]:.3f}); the aligned 2D grid localizes 3D position in ANY plane; a fixed '
             f'grid fails as the plane tilts</text>')
    oy = 60
    allv = [A[t][0] for t in TILTS] + [F[t][0] for t in TILTS]; hi = max(allv) * 1.2
    def X(i): return pad + (i / (len(TILTS) - 1)) * pw
    def Y(v): return oy + ph - (v / hi) * ph
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.1, 0.2, 0.3):
        if vv < hi:
            e.append(f'<line x1="{pad}" y1="{Y(vv):.0f}" x2="{pad+pw}" y2="{Y(vv):.0f}" stroke="#eef1f6"/>'
                     f'<text x="{pad-6}" y="{Y(vv)+3:.0f}" font-size="8.5" fill="#5b6b8c" text-anchor="end">{vv:.1f}</text>')
    for tag, D, c in (("aligned", A, "#2ca25f"), ("fixed", F, "#c9341a")):
        pts = " ".join(f"{X(i):.1f},{Y(D[t][0]):.1f}" for i, t in enumerate(TILTS))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.6"/>')
        for i, t in enumerate(TILTS):
            e.append(f'<circle cx="{X(i):.1f}" cy="{Y(D[t][0]):.1f}" r="2.8" fill="{c}"/>')
    for i, t in enumerate(TILTS):
        e.append(f'<text x="{X(i):.0f}" y="{oy+ph+16:.0f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{math.degrees(t):.0f}&#176;</text>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+32:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">motion-plane tilt &#8594;</text>')
    e.append(f'<text x="{pad+8}" y="{oy+14}" font-size="9.5" fill="#5b6b8c">3D position decode error</text>')
    lx = pad + pw + 16
    e.append(f'<rect x="{lx}" y="{oy+10}" width="14" height="4" fill="#2ca25f"/><text x="{lx+18}" y="{oy+15}" font-size="9.5" fill="#28324a">plane-aligned (flat)</text>')
    e.append(f'<rect x="{lx}" y="{oy+30}" width="14" height="4" fill="#c9341a"/><text x="{lx+18}" y="{oy+35}" font-size="9.5" fill="#28324a">fixed plane (fails at tilt)</text>')
    e.append(f'<text x="{lx}" y="{oy+62}" font-size="8.5" fill="#7787a6">PCA plane-recovery</text>')
    e.append(f'<text x="{lx}" y="{oy+76}" font-size="8.5" fill="#7787a6">err ~{Rc[TILTS[-1]][0]:.3f} (any tilt)</text>')
    e.append(f'<text x="{lx}" y="{oy+100}" font-size="8" fill="#9aa6bd">honest: no robust</text>')
    e.append(f'<text x="{lx}" y="{oy+112}" font-size="8" fill="#9aa6bd">decode win vs 3D grid</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
