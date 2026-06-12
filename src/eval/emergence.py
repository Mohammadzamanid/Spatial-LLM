"""
src/eval/emergence.py

Do neuroscience signatures EMERGE from the architecture (as the 7±2 working-memory limit did
from theta-gamma)? We pre-train the cortex on self-supervised path integration (PLACE-cell
prediction — NO periodic/grid structure imposed) and then measure, purely from the trained
units:

  1. GRID CELLS — spatial rate maps of hidden units over a 2D arena; count firing fields and
     measure lattice symmetry (4-fold square vs 6-fold hexagonal) + classic gridness (60°
     autocorrelogram symmetry). Periodic grid fields are the landmark emergent result of
     path-integration learning (Hafting 2005; Banino 2018; Cueva & Wei 2018).
  2. PATH-INTEGRATION DRIFT — decode position and test (a) distance COMPRESSION (PI
     systematically under-estimates how far you are) and (b) error growth with distance
     (scalar variability / Weber's law). Both are documented PI biases.
  3. HEAD-DIRECTION TUNING — is the conjunctive code directionally tuned (ring-attractor HD
     cells; Taube 1990)?

Writes results/emergence.json and (for the grid cells) results/emergence_gridcells.svg.
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import TrajectoryCortex


# ----------------------------------------------------------------------------- data
def walks_2d(n, T, seed, speed=(0.2, 0.8)):
    g = torch.Generator().manual_seed(seed)
    H = torch.rand(n, T, generator=g) * 2 * math.pi
    S = torch.rand(n, T, generator=g) * (speed[1] - speed[0]) + speed[0]
    V = torch.zeros(n, T)
    x = (S * H.cos()).sum(1); y = (S * H.sin()).sum(1)
    return H, S, V, torch.stack([x, y], -1)


def place_code(pos3, centers, sigma):
    d2 = ((pos3.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2))


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-9)).item()


# ------------------------------------------------------------------- train + capture
def train_cortex(epochs=100, dim=64, seed=0, topology="square", constrained=False):
    torch.manual_seed(seed)
    cx = TrajectoryCortex(embed_dim=dim, task="pathint", length_norm=False, topology=topology,
                          constrained_velocity=constrained)
    cg = torch.Generator().manual_seed(0)
    centers = torch.rand(128, 3, generator=cg) * 8 - 4               # bounded place cells, env ±4
    head = nn.Linear(dim, 128)
    opt = torch.optim.Adam(list(cx.parameters()) + list(head.parameters()), lr=3e-3)
    mse = nn.MSELoss()
    for _ in range(epochs):
        for T in (6, 8, 10, 12):
            H, S, V, xy = walks_2d(384, T, 100 + T)
            pos3 = torch.cat([xy, torch.zeros(xy.shape[0], 1)], -1)
            opt.zero_grad()
            mse(head(cx.encode(H, S, V)), place_code(pos3, centers, 1.0)).backward()
            opt.step()
    cx.eval()
    return cx


@torch.no_grad()
def capture(cx, H, S, V):
    if getattr(cx, "constrained", False):
        # the grid-cell population IS the per-module bump activity (B, K*M)
        v2d = torch.stack([S * H.cos(), S * H.sin()], dim=-1)
        _, cells = cx.integrator(v2d, return_cells=True)
        return cells, cx.encode(H, S, V), None
    B, T = H.shape
    step = (cx.conjunctive(H.reshape(B * T), S.reshape(B * T)).view(B, T, -1)
            + cx.vert(V.reshape(B * T, 1)).view(B, T, -1))
    u = torch.zeros(B, cx.integrator.N)
    for t in range(T):
        u = u + cx.integrator.vel_to_sheet(step[:, t])
        for _ in range(cx.integrator.settle):
            u = u + 0.1 * F.linear(torch.tanh(u), cx.integrator.W)
    h = cx.encode(H, S, V)
    return u, h, step


# ------------------------------------------------------------------------- rate maps
def rate_maps(pos2, acts, G=24, R=3.0, min_count=3):
    ix = ((pos2[:, 0] + R) / (2 * R) * G).long().clamp(0, G - 1)
    iy = ((pos2[:, 1] + R) / (2 * R) * G).long().clamp(0, G - 1)
    flat = iy * G + ix
    U = acts.shape[1]
    summ = torch.zeros(G * G, U).index_add_(0, flat, acts)
    cnt = torch.zeros(G * G).index_add_(0, flat, torch.ones(acts.shape[0]))
    mean = summ / cnt.clamp(min=1).unsqueeze(1)
    mean[cnt < min_count] = float("nan")
    return mean.t().reshape(U, G, G), cnt.reshape(G, G)


def _autocorr(rm):
    valid = ~torch.isnan(rm)
    r = torch.where(valid, rm, torch.zeros_like(rm))
    r = r - r[valid].mean()
    r = torch.where(valid, r, torch.zeros_like(rm))
    G = rm.shape[-1]
    P = F.pad(r, (0, G, 0, G))
    Fr = torch.fft.fft2(P)
    ac = torch.fft.ifft2(Fr * Fr.conj()).real
    ac = torch.fft.fftshift(ac)[1:, 1:]
    return ac / (ac.abs().max() + 1e-9)


def _rotate(img, deg):
    th = math.radians(deg)
    theta = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                          [math.sin(th), math.cos(th), 0.0]]).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, *img.shape), align_corners=False)
    return F.grid_sample(img.unsqueeze(0).unsqueeze(0), grid, align_corners=False)[0, 0]


def _npeaks(rm):
    a = torch.where(torch.isnan(rm), rm.new_full((), -1e9), rm)
    mx = F.max_pool2d(a.unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0]
    valid = ~torch.isnan(rm)
    if not valid.any():
        return 0
    thr = a[valid].float().quantile(0.80)
    return int(((a == mx) & (a > thr) & valid).sum())


def grid_stats(rm):
    """(gridness_hex, sym4_square, sym6_hex, periodicity, n_fields, autocorrelogram)."""
    ac = _autocorr(rm)
    G2 = ac.shape[0]; c = G2 // 2
    yy, xx = torch.meshgrid(torch.arange(G2) - c, torch.arange(G2) - c, indexing="ij")
    rad = torch.sqrt((xx ** 2 + yy ** 2).float())
    rmax = G2 * 0.5
    annulus = (rad > rmax * 0.18) & (rad < rmax * 0.95)
    base = ac[annulus]
    cor = {d: _pearson(base, _rotate(ac, d)[annulus]).item() for d in (30, 60, 90, 120, 150, 180)}
    # standard gridness: hexagonal if 60°/120° symmetry beats 30°/90°/150°.
    # (NB: never use 180° — an autocorrelogram is centrally symmetric, so c180≈1 for ANY map.)
    gridness = min(cor[60], cor[120]) - max(cor[30], cor[90], cor[150])
    sym6 = (cor[60] + cor[120]) / 2          # 6-fold (hexagonal) score
    sym4 = cor[90]                            # 4-fold (square) score
    periodicity = ac[annulus].max().item()
    return gridness, sym4, sym6, periodicity, _npeaks(rm), ac


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


# --------------------------------------------------------------------------- svg out
def _cmap(v):
    st = [(0.0, (68, 1, 84)), (0.5, (33, 144, 141)), (1.0, (253, 231, 37))]
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(st) - 1):
        a, b = st[i], st[i + 1]
        if v <= b[0]:
            f = (v - a[0]) / (b[0] - a[0] + 1e-9)
            cc = [round(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3)]
            return f"#{cc[0]:02x}{cc[1]:02x}{cc[2]:02x}"
    return "#fde725"


def _heat_svg(x, y, arr, cell):
    a = arr.clone(); valid = ~torch.isnan(a)
    lo = a[valid].min(); hi = a[valid].max(); s = ""
    for r in range(a.shape[0]):
        for cc in range(a.shape[1]):
            v = a[r, cc]
            col = "#161b29" if torch.isnan(v) else _cmap(((v - lo) / (hi - lo + 1e-9)).item())
            s += f'<rect x="{x+cc*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="{col}"/>'
    return s


def grid_svg(top, out="results/emergence_gridcells.svg", topology="square"):
    cols = len(top); cw, pad = 200, 20
    W = pad + cols * cw; Hh = 372
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    desc = ("velocity-driven hex grid modules" if topology == "hexvel" else f"{topology} torus")
    e.append(f'<text x="20" y="28" font-size="18" font-weight="800" fill="#0b1324">'
             f'Emergent periodic GRID fields — {desc} (measured symmetry per unit below)</text>')
    e.append('<text x="20" y="48" font-size="12" fill="#5b6b8c">hidden units of a cortex trained '
             'only to predict bounded PLACE cells — multi-peak periodic fields emerged on their own.</text>')
    for i, (uid, gr, s4, s6, per, npk, rm, ac) in enumerate(top):
        x = pad + i * cw
        e.append(f'<text x="{x}" y="84" font-size="12.5" font-weight="700" fill="#28324a">unit {uid} · {npk} fields</text>')
        e.append(_heat_svg(x, 92, rm, 150 / rm.shape[0]))
        e.append(f'<text x="{x}" y="262" font-size="11" fill="#5b6b8c">rate map (firing vs x,y)</text>')
        e.append(_heat_svg(x, 272, ac, 70 / ac.shape[0]))
        e.append(f'<text x="{x+78}" y="290" font-size="11" fill="#28324a">periodicity {per:.2f}</text>')
        e.append(f'<text x="{x+78}" y="306" font-size="11" fill="#28324a">4-fold {s4:.2f}</text>')
        e.append(f'<text x="{x+78}" y="322" font-size="11" fill="#28324a">6-fold {s6:.2f}</text>')
        e.append(f'<text x="{x}" y="358" font-size="11" fill="#5b6b8c">autocorrelogram</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    return out


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topology", choices=["square", "hex"], default="square",
                    help="attractor torus: square (default) or hex/twisted (Guanella 2007)")
    ap.add_argument("--constrained", action="store_true",
                    help="velocity-driven hexagonal grid MODULES (fixed gains, learned readout) — "
                         "the Burak-Fiete/Guanella construction; predicts true hexagonal grids")
    a = ap.parse_args()
    suf = "_hexvel" if a.constrained else ("" if a.topology == "square" else f"_{a.topology}")
    print(f"training cortex (self-supervised PLACE-cell path integration, {a.epochs} ep, "
          f"topology={a.topology}, constrained_velocity={a.constrained})...", flush=True)
    cx = train_cortex(a.epochs, seed=a.seed, topology=a.topology, constrained=a.constrained)

    H, S, V, xy = walks_2d(20000, 9, 7)
    u, h, step = capture(cx, H, S, V)
    print(f"captured {xy.shape[0]} positions; building rate maps...\n", flush=True)

    out = {}
    print("=== 1. GRID CELLS (rate maps of path-integrating units) ===", flush=True)
    for name, acts in [("sheet", u), ("summary_h", h)]:
        rms, cover = rate_maps(xy, acts, G=a.grid, R=a.R)
        stats = []
        for ui in range(rms.shape[0]):
            if torch.isnan(rms[ui]).float().mean() > 0.5:
                continue
            gr, s4, s6, per, npk, ac = grid_stats(rms[ui])
            stats.append((ui, gr, s4, s6, per, npk, rms[ui], ac))
        stats.sort(key=lambda z: z[1], reverse=True)               # highest gridness first
        n = len(stats)
        n_periodic = sum(1 for z in stats if z[4] > 0.30)
        n_multi = sum(1 for z in stats if z[5] >= 3)
        n_hex = sum(1 for z in stats if z[1] > 0 and z[4] > 0.30)            # gridness>0 = hexagonal
        n_sq = sum(1 for z in stats if z[1] <= 0 and z[2] > z[3] and z[4] > 0.30)
        mean_fields = round(sum(z[5] for z in stats) / max(n, 1), 1)
        mean_grid = round(sum(z[1] for z in stats) / max(n, 1), 3)
        out[name] = {"n_units": n, "n_periodic": n_periodic, "n_multifield": n_multi,
                     "mean_fields": mean_fields, "n_square_lattice": n_sq, "n_hex_lattice": n_hex,
                     "mean_gridness": mean_grid,
                     "top_gridness": round(max((z[1] for z in stats), default=0), 3)}
        print(f"[{name}] units={n}  multi-field(≥3)={n_multi} ({100*n_multi//max(n,1)}%)  "
              f"mean fields/unit={mean_fields}  mean gridness={mean_grid}  "
              f"HEX(gridness>0)={n_hex}  square={n_sq}", flush=True)
        if name == "sheet":
            top = [(z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]) for z in stats[:6]]
            out["gridcell_svg"] = grid_svg(top, out=f"results/emergence_gridcells{suf}.svg",
                                           topology=("hexvel" if a.constrained else a.topology))
            print(f"  wrote {out['gridcell_svg']}", flush=True)

    # ---- 2. path-integration drift: distance compression + Weber's law ----
    print("\n=== 2. PATH-INTEGRATION DRIFT (decode position, error vs distance) ===", flush=True)
    probe = nn.Sequential(nn.Linear(h.shape[1], 128), nn.ReLU(), nn.Linear(128, 2))
    po = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(2500):
        po.zero_grad(); F.mse_loss(probe(h), xy).backward(); po.step()
    Ht, St, Vt, xyt = walks_2d(8000, 9, 9)
    ut, ht, _ = capture(cx, Ht, St, Vt)
    with torch.no_grad():
        pred = probe(ht)
    dist = xyt.norm(dim=1); pdist = pred.norm(dim=1); err = (pred - xyt).norm(dim=1)
    k = ((pdist * dist).sum() / (dist * dist).sum()).item()        # pred_dist ≈ k·true_dist
    dcorr = 0.5 * (_corr(pred[:, 0], xyt[:, 0]) + _corr(pred[:, 1], xyt[:, 1]))
    bins = torch.linspace(dist.min(), dist.quantile(0.95), 6)
    weber = []
    for i in range(len(bins) - 1):
        m = (dist >= bins[i]) & (dist < bins[i + 1])
        if m.sum() > 30:
            weber.append({"dist": round(dist[m].mean().item(), 2),
                          "error": round(err[m].mean().item(), 3),
                          "error/dist": round((err[m].mean() / dist[m].mean()).item(), 3)})
    out["pi_drift"] = {"decode_corr": round(dcorr, 3), "distance_compression_k": round(k, 3),
                       "weber_bins": weber}
    print(f"  position decode corr={dcorr:.2f}", flush=True)
    print(f"  distance COMPRESSION: pred ≈ {k:.2f}×true  ->  "
          f"path integration {'UNDER' if k < 1 else 'over'}-estimates distance "
          f"(biological PI homing-vector bias)", flush=True)
    print("  error vs distance:", "  ".join(f"d≈{w['dist']}:err={w['error']}" for w in weber), flush=True)

    # ---- 3. head-direction tuning ----
    print("\n=== 3. HEAD-DIRECTION TUNING (conjunctive code) ===", flush=True)
    ang = torch.linspace(0, 2 * math.pi, 240)
    with torch.no_grad():
        conj = cx.conjunctive(ang, torch.full_like(ang, 0.6))
    vs = []
    for uidx in range(conj.shape[1]):
        r = conj[:, uidx]; r = (r - r.min()) / (r.max() - r.min() + 1e-9)
        vs.append((((r * torch.exp(1j * ang)).sum().abs()) / (r.sum() + 1e-9)).item())
    vs = torch.tensor(vs)
    out["head_direction"] = {"mean_vector_strength": round(vs.mean().item(), 3),
                             "frac_tuned(>0.4)": round((vs > 0.4).float().mean().item(), 3)}
    print(f"  conjunctive units: mean tuning vector-strength={vs.mean():.2f}, "
          f"{100*(vs>0.4).float().mean():.0f}% directionally tuned (ring-attractor HD code)", flush=True)

    out["topology"] = a.topology
    with open(f"results/emergence{suf}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/emergence{suf}.json", flush=True)


if __name__ == "__main__":
    main()
