"""
src/eval/local_3d_order.py

LOCAL 3D ORDER, NOT A GLOBAL LATTICE — the bat 3D grid-cell regime.

Freely-flying bats have 3D head-direction cells, 3D border cells, and 3D grid-like multi-field neurons in
MEC, but the 3D grid cells show LOCAL order (regular nearest-neighbor field spacing) WITHOUT a global 3D
lattice (no long-range periodicity / FCC-like crystal) (Nature, bats). This complements `plane_of_motion.py`
(which implements the plane-aligned 2D-grid scheme): here we make the "local order, not global lattice"
claim *measurable*, so the repo's 3D story is characterized rather than assumed.

We score a set of 3D field centers on two independent axes:
  - LOCAL ORDER = 1 - CV(nearest-neighbor distance): is the spacing regular? (high for a lattice AND for a
    minimum-distance "blue-noise" packing; low for random points).
  - GLOBAL LATTICE = max structure factor S(q)/N over a Bragg-region q-scan: is there long-range periodic
    order? (S(q) = |sum exp(i q.r)|^2 / N; ~1 at a Bragg peak for a lattice, only a broad liquid peak for
    blue-noise, flat for random). Periodic (toroidal) distances are used so the metrics are boundary-free.

The result: a LOCAL-ORDER (blue-noise) field code has HIGH local order but LOW global lattice — exactly the
bat regime — and is cleanly separable from a true 3D lattice (high on both) and from random points (low on
both). So "local order without a global lattice" is a well-defined, measurable third regime.

Multi-seed, mean +/- 95% CI. Writes results/local_3d_order.json + .svg.

    python -m src.eval.local_3d_order --seeds 5
"""
import argparse
import json
import math
import os

import torch

N = 125; KIND = ["lattice", "local_order", "random"]


def periodic_cdist(a, b):
    d = (a.unsqueeze(1) - b.unsqueeze(0)).abs(); d = torch.minimum(d, 1 - d)
    return d.norm(dim=2)


def lattice(side, jitter, gen):
    g = torch.arange(side).float() / side
    p = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return (p + jitter * torch.randn(p.shape, generator=gen)) % 1.0


def blue_noise(n, rmin, gen, max_tries=80000):
    pts = torch.empty(0, 3); t = 0
    while pts.shape[0] < n and t < max_tries:
        c = torch.rand(1, 3, generator=gen)
        if pts.shape[0] == 0 or periodic_cdist(c, pts).min().item() > rmin:
            pts = torch.cat([pts, c], 0)
        t += 1
    return pts


def random_pts(n, gen):
    return torch.rand(n, 3, generator=gen)


def local_order(pts):
    d = periodic_cdist(pts, pts); d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return 1.0 - (nn.std(unbiased=True) / nn.mean()).item()


def global_lattice(pts, qlo=18.0, qhi=50.0, nq=140):
    """max structure factor over a Bragg-region q-scan (skip low q where S->N artifactually)."""
    dirs = [torch.tensor([1., 0, 0]), torch.tensor([0, 1., 0]), torch.tensor([0, 0, 1.]),
            torch.tensor([1., 1, 0]) / 2 ** .5, torch.tensor([1., 1, 1]) / 3 ** .5]
    mags = torch.linspace(qlo, qhi, nq); best = 0.0
    for d in dirs:
        for m in mags:
            ph = pts @ (m * d)
            best = max(best, ((ph.cos().sum() ** 2 + ph.sin().sum() ** 2) / pts.shape[0]).item())
    return best / pts.shape[0]                       # normalise to [0,1] (= |sum exp|^2 / N^2; 1 at a Bragg peak)


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed)
    sets = {"lattice": lattice(5, 0.012, gen), "local_order": blue_noise(N, 0.165, gen), "random": random_pts(N, gen)}
    return {k: {"local": local_order(p), "global": global_lattice(p)} for k, p in sets.items()}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {k: {m: ci([p[k][m] for p in per]) for m in ("local", "global")} for k in KIND}

    print(f"\nLOCAL 3D ORDER, NOT A GLOBAL LATTICE — the bat 3D grid regime (n={a.seeds}; mean ± 95% CI)\n" + "=" * 78, flush=True)
    print(f"    {'3D field code':>16} | {'LOCAL order (1-CV NN)':>22} | {'GLOBAL lattice (max S/N)':>24}", flush=True)
    lab = {"lattice": "global 3D lattice", "local_order": "LOCAL-order (bat-like)", "random": "random"}
    for k in KIND:
        print(f"    {lab[k]:>16} | {agg[k]['local'][0]:>20.2f}   | {agg[k]['global'][0]:>24.2f}", flush=True)
    lo = agg["local_order"]; la = agg["lattice"]; rd = agg["random"]
    print(f"\n  -> a LOCAL-ORDER 3D field code has HIGH local order ({lo['local'][0]:.2f}) but LOW global lattice "
          f"({lo['global'][0]:.2f}) -- regular nearest-neighbor spacing with NO long-range periodicity, exactly "
          f"the bat 3D grid regime. It is cleanly separable from a true 3D lattice (high BOTH: {la['local'][0]:.2f}/"
          f"{la['global'][0]:.2f}) and from random points (low both: {rd['local'][0]:.2f}/{rd['global'][0]:.2f}). "
          f"'Local order without a global lattice' is a well-defined, measurable regime -- so the repo's 3D "
          f"story is the bat-faithful one, not a naive cubic lattice.", flush=True)

    out = {"n_seeds": a.seeds, "n_points": N, "results": {k: agg[k] for k in KIND}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/local_3d_order.json", "w"), indent=2)
    svg(agg, "results/local_3d_order.svg")
    print("\nwrote results/local_3d_order.json and results/local_3d_order.svg", flush=True)


def svg(agg, out):
    pad = 64; sz = 300; W = pad + sz + 200; H = 70 + sz + 30
    col = {"lattice": "#3182bd", "local_order": "#2ca25f", "random": "#c9341a"}
    lab = {"lattice": "global lattice", "local_order": "LOCAL-order (bat)", "random": "random"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Local 3D order without a global lattice: the bat grid regime</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">two independent axes: regular spacing '
             '(local) vs long-range periodicity (global lattice)</text>')
    ox = pad; oy = 58
    def X(v): return ox + v * sz
    def Y(v): return oy + sz - v * sz
    e.append(f'<rect x="{ox}" y="{oy}" width="{sz}" height="{sz}" fill="none" stroke="#33415c"/>')
    for v in (0.0, 0.5, 1.0):
        e.append(f'<text x="{ox-6}" y="{Y(v)+3:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{v:.1f}</text>')
        e.append(f'<text x="{X(v):.0f}" y="{oy+sz+14:.0f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{v:.1f}</text>')
    e.append(f'<text x="{ox+sz/2:.0f}" y="{oy+sz+28:.0f}" font-size="10" fill="#28324a" text-anchor="middle">LOCAL order (regular spacing) &#8594;</text>')
    e.append(f'<text x="{ox-40}" y="{oy+sz/2:.0f}" font-size="10" fill="#28324a" text-anchor="middle" transform="rotate(-90 {ox-40} {oy+sz/2:.0f})">GLOBAL lattice (periodicity) &#8594;</text>')
    for k in KIND:
        lx, gl = agg[k]["local"][0], agg[k]["global"][0]
        e.append(f'<circle cx="{X(lx):.1f}" cy="{Y(gl):.1f}" r="8" fill="{col[k]}" opacity="0.9"/>')
        e.append(f'<text x="{X(lx)+12:.1f}" y="{Y(gl)+4:.1f}" font-size="10" font-weight="700" fill="{col[k]}">{lab[k]}</text>')
    e.append(f'<text x="{X(0.5):.0f}" y="{Y(0.93):.0f}" font-size="8.5" fill="#7787a6" text-anchor="middle">crystal (high both)</text>')
    e.append(f'<text x="{X(0.85):.0f}" y="{Y(0.18):.0f}" font-size="8.5" fill="#2ca25f" text-anchor="middle">bat: local, no lattice</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
