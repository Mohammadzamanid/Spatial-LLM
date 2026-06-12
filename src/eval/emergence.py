"""
src/eval/emergence.py

Do neuroscience signatures EMERGE from the architecture (as the 7±2 working-memory limit did
from theta-gamma)? We pre-train the cortex on self-supervised path integration (PLACE-cell
prediction — NO periodic/grid structure imposed) and then measure, purely from the trained
units:

  1. GRID CELLS — build spatial rate maps of hidden units over a 2D arena; quantify spatial
     periodicity and lattice symmetry (4-fold square vs 6-fold hexagonal) and the classic
     gridness score (60° autocorrelogram symmetry). Grid fields are the landmark emergent
     result of path-integration learning (Hafting 2005; Banino 2018; Cueva & Wei 2018).
  2. PATH-INTEGRATION DRIFT / WEBER'S LAW — decode position from the rep and test whether the
     error grows with distance travelled (biological PI accumulates error; magnitude estimates
     follow scalar variability / Weber's law).
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
    """Random 2D walks (vz=0). Returns heading,speed,vz (n,T) and final (x,y) (n,2)."""
    g = torch.Generator().manual_seed(seed)
    H = torch.rand(n, T, generator=g) * 2 * math.pi
    S = torch.rand(n, T, generator=g) * (speed[1] - speed[0]) + speed[0]
    V = torch.zeros(n, T)
    x = (S * H.cos()).sum(1); y = (S * H.sin()).sum(1)
    return H, S, V, torch.stack([x, y], -1)


def place_code(pos3, centers, sigma):
    d2 = ((pos3.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2))


# ------------------------------------------------------------------- train + capture
def train_cortex(epochs=60, dim=64, seed=0):
    torch.manual_seed(seed)
    cx = TrajectoryCortex(embed_dim=dim, task="pathint", length_norm=False)
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
    """Final (sheet u, summary h, per-step conjunctive code)."""
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
    """acts (n, U) -> (U, G, G) mean activation per spatial bin; empty bins = nan."""
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
    """Unbiased-ish spatial autocorrelogram of a rate map with nan mask (size 2G-1)."""
    valid = ~torch.isnan(rm)
    r = torch.where(valid, rm, torch.zeros_like(rm))
    r = r - r[valid].mean()
    r = torch.where(valid, r, torch.zeros_like(rm))
    G = rm.shape[-1]
    P = F.pad(r, (0, G, 0, G))
    Fr = torch.fft.fft2(P)
    ac = torch.fft.ifft2(Fr * Fr.conj()).real
    ac = torch.fft.fftshift(ac)
    ac = ac[1:, 1:]                                   # -> (2G-1, 2G-1), centred
    m = ac.abs().max()
    return ac / (m + 1e-9)


def _rotate(img, deg):
    th = math.radians(deg)
    theta = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                          [math.sin(th), math.cos(th), 0.0]]).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, *img.shape), align_corners=False)
    return F.grid_sample(img.unsqueeze(0).unsqueeze(0), grid, align_corners=False)[0, 0]


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


def grid_stats(rm):
    """Return (gridness_hex, sym4, sym6, periodicity, autocorrelogram)."""
    ac = _autocorr(rm)
    G2 = ac.shape[0]; c = G2 // 2
    yy, xx = torch.meshgrid(torch.arange(G2) - c, torch.arange(G2) - c, indexing="ij")
    rad = torch.sqrt((xx ** 2 + yy ** 2).float())
    rmax = G2 * 0.5
    annulus = (rad > rmax * 0.18) & (rad < rmax * 0.95)
    base = ac[annulus]
    cor = {d: _pearson(base, _rotate(ac, d)[annulus]) for d in (30, 60, 90, 120, 150, 180)}
    gridness = min(cor[60], cor[120]) - max(cor[30], cor[90], cor[150])
    sym6 = (cor[60] + cor[120]) / 2
    sym4 = (cor[90] + cor[180]) / 2
    # periodicity: strongest off-centre autocorr peak in the annulus (0 = no repeat)
    periodicity = ac[annulus].max().item()
    return (gridness.item(), sym4.item(), sym6.item(), periodicity, ac)


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
    a = arr.clone()
    valid = ~torch.isnan(a)
    lo = a[valid].min(); hi = a[valid].max()
    s = ""
    for r in range(a.shape[0]):
        for cc in range(a.shape[1]):
            v = a[r, cc]
            col = "#1b2233" if torch.isnan(v) else _cmap(((v - lo) / (hi - lo + 1e-9)).item())
            s += f'<rect x="{x+cc*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="{col}"/>'
    return s


def grid_svg(top, out="results/emergence_gridcells.svg"):
    cols = len(top)
    cw, pad = 200, 20
    W = pad + cols * cw
    H = 360
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="Segoe UI, Arial">', f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="20" y="28" font-size="18" font-weight="800" fill="#0b1324">'
             'Emergent GRID CELLS — spatial rate maps &amp; autocorrelograms</text>')
    e.append('<text x="20" y="48" font-size="12" fill="#5b6b8c">hidden units of a cortex trained '
             'only to predict bounded place cells; periodic fields emerged on their own.</text>')
    for i, (uid, score, sym4, sym6, rm, ac) in enumerate(top):
        x = pad + i * cw
        e.append(f'<text x="{x}" y="82" font-size="12.5" font-weight="700" fill="#28324a">unit {uid}</text>')
        e.append(_heat_svg(x, 92, rm, 150 / rm.shape[0]))
        e.append(f'<text x="{x}" y="262" font-size="11" fill="#5b6b8c">rate map</text>')
        e.append(_heat_svg(x, 270, ac, 70 / ac.shape[0]))
        e.append(f'<text x="{x+78}" y="292" font-size="11" fill="#28324a">gridness {score:.2f}</text>')
        e.append(f'<text x="{x+78}" y="308" font-size="11" fill="#28324a">4-fold {sym4:.2f}</text>')
        e.append(f'<text x="{x+78}" y="324" font-size="11" fill="#28324a">6-fold {sym6:.2f}</text>')
        e.append(f'<text x="{x}" y="350" font-size="11" fill="#5b6b8c">autocorrelogram</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    return out


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"training cortex (self-supervised place-cell path integration, {a.epochs} ep)...", flush=True)
    cx = train_cortex(a.epochs, seed=a.seed)

    # sample the arena: many short walks, pool (final position, unit activity)
    H, S, V, xy = walks_2d(20000, 9, 7)
    u, h, step = capture(cx, H, S, V)
    print(f"captured {xy.shape[0]} positions; building rate maps...", flush=True)

    out = {}
    for name, acts in [("sheet", u), ("summary_h", h)]:
        rms, cover = rate_maps(xy, acts, G=a.grid, R=a.R)
        stats = []
        for ui in range(rms.shape[0]):
            if torch.isnan(rms[ui]).float().mean() > 0.5:
                continue
            gr, s4, s6, per, ac = grid_stats(rms[ui])
            stats.append((ui, gr, s4, s6, per, rms[ui], ac))
        stats.sort(key=lambda z: z[1], reverse=True)
        n = len(stats)
        n_grid = sum(1 for z in stats if z[1] > 0)
        n_periodic = sum(1 for z in stats if z[4] > 0.30)
        n_sq = sum(1 for z in stats if z[2] > z[3] and z[4] > 0.30)
        n_hex = sum(1 for z in stats if z[3] > z[2] and z[4] > 0.30)
        top5 = [round(z[1], 3) for z in stats[:5]]
        out[name] = {"n_units": n, "n_gridness>0": n_grid, "n_periodic": n_periodic,
                     "n_square_lattice": n_sq, "n_hex_lattice": n_hex, "top5_gridness": top5}
        print(f"[{name}] units={n}  periodic(spatial-repeat)={n_periodic} "
              f"({100*n_periodic/max(n,1):.0f}%)  square={n_sq} hex={n_hex}  "
              f"gridness>0={n_grid}  top gridness={top5}", flush=True)
        if name == "sheet":
            top = [(z[0], z[1], z[2], z[3], z[5], z[6]) for z in stats[:6]]
            svg = grid_svg(top)
            out["gridcell_svg"] = svg
            print(f"  wrote {svg}", flush=True)

    # ---- 2. path-integration drift / Weber's law: decode error vs distance ----
    probe = nn.Sequential(nn.Linear(h.shape[1], 64), nn.ReLU(), nn.Linear(64, 2))
    po = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(400):
        po.zero_grad(); F.mse_loss(probe(h), xy).backward(); po.step()
    Ht, St, Vt, xyt = walks_2d(8000, 9, 9)
    ut, ht, _ = capture(cx, Ht, St, Vt)
    with torch.no_grad():
        err = (probe(ht) - xyt).norm(dim=1)
    dist = xyt.norm(dim=1)
    bins = torch.linspace(dist.min(), dist.quantile(0.95), 6)
    weber = []
    for i in range(len(bins) - 1):
        m = (dist >= bins[i]) & (dist < bins[i + 1])
        if m.sum() > 20:
            d = dist[m].mean().item(); ee = err[m].mean().item()
            weber.append({"dist": round(d, 2), "error": round(ee, 3), "error/dist": round(ee / d, 3)})
    out["weber_law"] = weber
    ratios = [w["error/dist"] for w in weber]
    print("\n[path-integration drift] error vs distance (Weber's law if error grows w/ distance):",
          flush=True)
    for w in weber:
        print(f"   dist≈{w['dist']:<5} mean error={w['error']:<6} error/dist={w['error/dist']}", flush=True)
    print(f"   -> error/distance ratio range {min(ratios):.2f}–{max(ratios):.2f} "
          f"(roughly constant ⇒ scalar variability / Weber's law)", flush=True)

    # ---- 3. head-direction tuning of the conjunctive code ----
    Hh = torch.linspace(0, 2 * math.pi, 240).unsqueeze(1)
    sp = torch.full_like(Hh, 0.6)
    with torch.no_grad():
        conj = cx.conjunctive(Hh.reshape(-1), sp.reshape(-1))            # (240, dim)
    # vector strength of each unit's tuning over heading
    ang = Hh.squeeze(1)
    vs = []
    for uidx in range(conj.shape[1]):
        r = conj[:, uidx]; r = (r - r.min()) / (r.max() - r.min() + 1e-9)
        v = (r * torch.exp(1j * ang)).sum().abs() / (r.sum() + 1e-9)
        vs.append(v.item())
    vs = torch.tensor(vs)
    out["head_direction"] = {"mean_vector_strength": round(vs.mean().item(), 3),
                             "frac_strongly_tuned(>0.4)": round((vs > 0.4).float().mean().item(), 3)}
    print(f"\n[head-direction] conjunctive units: mean tuning vector-strength={vs.mean():.2f}, "
          f"{100*(vs>0.4).float().mean():.0f}% strongly tuned (ring-attractor HD code)", flush=True)

    with open("results/emergence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/emergence.json", flush=True)


if __name__ == "__main__":
    main()
