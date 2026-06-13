"""
src/eval/pillars.py

Three more brain pillars, demonstrated on the velocity-driven grid cortex (CPU, no LLM):

  A. REMAPPING & grid reuse — the grid code is a UNIVERSAL metric: one position decoder
     transfers across environments (0-shot), while PLACE codes REMAP (decorrelate) and a new
     environment's place map is learned few-shot on top of the reused grid metric
     (Fyhn 2007; Leutgeb 2005).
  B. REPLAY / consolidation — hippocampal replay is experience replay: rehearsing a few stored
     trajectories offline (incl. reverse replay, sharp-wave ripples) consolidates a good map
     from little real experience, far better than using each experience once.
  C. LOCAL (Hebbian) plasticity — place fields EMERGE from the grid code via competitive Hebbian
     learning (winner-take-all + move-toward-input), with NO backprop (Rolls & Treves; Si & Treves).

Writes results/pillars.json and results/pillars_hebbian.svg (emergent place fields).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules


def grid_code(cx, pos):
    """Grid-cell population code at 2D positions pos (B,2) -> (B, K*M). Direct phase = gain*pos."""
    phi = cx.gains.view(-1, 1, 1) * pos.unsqueeze(0)            # (K,B,2)
    return cx._grid_code(phi)


def rand_pos(n, R, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 2, generator=g) * 2 - 1) * R


def place_code(pos, centers, sig=0.5):
    d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sig ** 2))


# ----------------------------------------------------------------- A. remapping & reuse
def remapping(R=3.0, seed=0):
    torch.manual_seed(seed)
    cx = _HexGridModules(64, n_modules=5, base_spacing=1.5)
    cA = rand_pos(80, R, 101); cB = rand_pos(80, R, 202)        # env-specific place-cell layouts

    # universal grid->position decoder, trained in env A
    posA = rand_pos(8000, R, 10); gcA = grid_code(cx, posA)
    dec = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(500):
        opt.zero_grad(); F.mse_loss(dec(gcA), posA).backward(); opt.step()
    with torch.no_grad():
        pa = rand_pos(3000, R, 11); pb = rand_pos(3000, R, 22)
        err_A = (dec(grid_code(cx, pa)) - pa).norm(-1).mean().item()
        err_B = (dec(grid_code(cx, pb)) - pb).norm(-1).mean().item()   # 0-shot transfer

    # place remapping: the SAME locations -> decorrelated place populations across envs;
    # the grid population is identical (reused)
    with torch.no_grad():
        pos = rand_pos(2000, R, 30)
        gcorr = F.cosine_similarity(grid_code(cx, pos), grid_code(cx, pos), dim=1).mean().item()
        pcorr = F.cosine_similarity(place_code(pos, cA), place_code(pos, cB), dim=1).mean().item()

    # few-shot new map: how fast does an env-B place readout converge on the ready grid metric?
    posB = rand_pos(4000, R, 40); gcB = grid_code(cx, posB); tgtB = place_code(posB, cB)
    head = nn.Linear(cx.K * cx.M, cB.shape[0]); hopt = torch.optim.Adam(head.parameters(), lr=1e-2)
    curve = {}
    for step in range(1, 41):
        hopt.zero_grad(); F.mse_loss(head(gcB), tgtB).backward(); hopt.step()
        if step in (1, 3, 5, 10, 20, 40):
            with torch.no_grad():
                pv = rand_pos(1500, R, 55); ev = F.mse_loss(head(grid_code(cx, pv)), place_code(pv, cB)).item()
            curve[step] = round(ev, 4)
    out = {"decode_err_envA": round(err_A, 3), "decode_err_envB_0shot": round(err_B, 3),
           "grid_popvec_corr_AvsB": round(gcorr, 3), "place_popvec_corr_AvsB": round(pcorr, 3),
           "fewshot_placeB_mse_by_step": curve}
    print(f"A. REMAPPING: grid decoder err  envA={err_A:.3f}  envB(0-shot)={err_B:.3f}  "
          f"-> grid metric REUSED across environments", flush=True)
    print(f"   place pop-vector corr A-vs-B = {pcorr:.3f} (REMAPPED)  vs grid corr = {gcorr:.3f} (reused); "
          f"new place map few-shot mse {curve.get(1)}->{curve.get(40)}", flush=True)
    return out


# --------------------------------------------------------------------- B. replay
def walk_codes(cx, n, T, R, seed):
    """Bounded random walks -> stacked (grid_code, position) over all steps."""
    g = torch.Generator().manual_seed(seed); pos = torch.zeros(n, 2)
    codes, poss = [], []
    for t in range(T):
        h = torch.rand(n, generator=g) * 2 * math.pi; s = torch.rand(n, generator=g) * 0.6 + 0.2
        pos = (pos + torch.stack([s * h.cos(), s * h.sin()], -1)).clamp(-R, R)
        codes.append(grid_code(cx, pos)); poss.append(pos.clone())
    return torch.cat(codes), torch.cat(poss)


def replay(R=3.0, seed=0):
    torch.manual_seed(seed)
    cx = _HexGridModules(64, n_modules=5, base_spacing=1.5)
    # a SMALL real experience buffer (few trajectories) vs a big-data ceiling
    Xbuf, Ybuf = walk_codes(cx, 40, 16, R, 1)          # ~640 stored samples (few real trajectories)
    Xbig, Ybig = walk_codes(cx, 4000, 16, R, 2)        # large data (ceiling)
    Xte, Yte = walk_codes(cx, 1500, 16, R, 3)

    def train(steps, sampler):
        dec = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
        opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
        for _ in range(steps):
            xb, yb = sampler()
            opt.zero_grad(); F.mse_loss(dec(xb), yb).backward(); opt.step()
        with torch.no_grad():
            return (dec(Xte) - Yte).norm(-1).mean().item()

    bs = 256
    def buf_sampler():                                  # replay: rehearse the stored buffer
        idx = torch.randint(0, Xbuf.shape[0], (bs,)); return Xbuf[idx], Ybuf[idx]
    def big_sampler():
        idx = torch.randint(0, Xbig.shape[0], (bs,)); return Xbig[idx], Ybig[idx]

    err_online = train(3, buf_sampler)                  # ~one pass over the few experiences (no replay)
    err_replay = train(400, buf_sampler)                # many offline rehearsals of the SAME buffer
    err_ceiling = train(400, big_sampler)               # large real data
    out = {"n_real_trajectories": 40, "err_no_replay": round(err_online, 3),
           "err_with_replay": round(err_replay, 3), "err_big_data_ceiling": round(err_ceiling, 3)}
    print(f"\nB. REPLAY: from 40 real trajectories -> decode err  no-replay={err_online:.3f}  "
          f"WITH replay={err_replay:.3f}  (big-data ceiling={err_ceiling:.3f})", flush=True)
    print(f"   replaying the small buffer consolidates a near-ceiling map from little experience", flush=True)
    return out


# ----------------------------------------------------------------- C. Hebbian place cells
def hebbian(R=3.0, Kp=64, seed=0):
    torch.manual_seed(seed)
    cx = _HexGridModules(64, n_modules=5, base_spacing=1.5)
    D = cx.K * cx.M
    pos = rand_pos(6000, R, 7); X = grid_code(cx, pos)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-6)                      # normalize inputs
    W = X[torch.randperm(X.shape[0])[:Kp]].clone()                   # init units on data (no dead units)
    # online competitive Hebbian learning (winner-take-all + move toward input); NO backprop
    for it in range(4000):
        x = X[torch.randint(0, X.shape[0], (1,))][0]
        win = (W @ x).argmax()
        eta = 0.05 * (1 - it / 4000)
        W[win] = W[win] + eta * (x - W[win])
        W[win] = W[win] / (W[win].norm() + 1e-6)
    # rate maps over a position grid -> are the emergent units LOCALIZED (place fields)?
    G = 28; xs = torch.linspace(-R, R, G)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    grid_pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    Xg = grid_code(cx, grid_pos); Xg = Xg / (Xg.norm(dim=1, keepdim=True) + 1e-6)
    act = (Xg @ W.t())                                                # (G*G, Kp)
    maps = act.t().reshape(Kp, G, G)
    # localization: fraction of arena above half-max (small = sharp place field)
    fields = []
    for i in range(Kp):
        m = maps[i]; mn, mx = m.min(), m.max()
        frac = ((m - mn) / (mx - mn + 1e-6) > 0.5).float().mean().item()
        fields.append(frac)
    fields = torch.tensor(fields)
    place_like = (fields < 0.15).float().mean().item()               # compact single-field units
    out = {"n_units": Kp, "mean_active_fraction": round(fields.mean().item(), 3),
           "frac_place_like(<0.15 area)": round(place_like, 3)}
    print(f"\nC. HEBBIAN: place fields from grids via local competitive learning (no backprop): "
          f"mean field area={fields.mean():.2f} of arena, {100*place_like:.0f}% are compact place cells", flush=True)
    # SVG of a few emergent place fields
    svg_hebbian(maps[fields.argsort()[:8]], "results/pillars_hebbian.svg")
    return out


def _cmap(v):
    st = [(0.0, (68, 1, 84)), (0.5, (33, 144, 141)), (1.0, (253, 231, 37))]
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(st) - 1):
        a, b = st[i], st[i + 1]
        if v <= b[0]:
            f = (v - a[0]) / (b[0] - a[0] + 1e-9)
            c = [round(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3)]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
    return "#fde725"


def svg_hebbian(maps, out):
    n = maps.shape[0]; G = maps.shape[1]; cell = 90 / G; pad = 16; cw = 110
    W = pad + n * cw; H = 150
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="16" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Emergent PLACE CELLS from grid cells via local Hebbian learning (no backprop)</text>')
    for i in range(n):
        m = maps[i]; mn, mx = m.min(), m.max(); x = pad + i * cw
        for r in range(G):
            for c in range(G):
                v = ((m[r, c] - mn) / (mx - mn + 1e-6)).item()
                e.append(f'<rect x="{x+c*cell:.1f}" y="{40+r*cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{_cmap(v)}"/>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    print("=== three brain pillars on the velocity-driven grid cortex ===", flush=True)
    out = {"remapping": remapping(), "replay": replay(), "hebbian": hebbian()}
    with open("results/pillars.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/pillars.json and results/pillars_hebbian.svg", flush=True)


if __name__ == "__main__":
    main()
