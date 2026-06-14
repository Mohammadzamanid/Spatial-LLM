"""
src/eval/embodiment.py

EMBODIMENT — ground the map in PERCEPTION: see the world, infer self-motion from optic flow.

So far the cortex was handed (heading, speed). The brain instead DERIVES self-motion from sensory
flow — optic flow across the retina — and path-integrates that. We give the agent a simple visual
world (landmarks), a retinal PANORAMA at each position, and a learned visual front-end that estimates
its velocity from how the panorama shifts between two frames (optic-flow egomotion). That
visually-estimated velocity drives the SAME grid cortex.

Tests: (1) the front-end recovers self-motion from vision; (2) the grid map path-integrates the
VISION-derived velocity and still localizes — position from what the agent SEES, not from hand-given
motion — close to the true-velocity ceiling. Writes results/embodiment.json + .svg.
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules

B_BINS = 36                                                          # panoramic photoreceptor bins


def grid_code(cx, pos):
    phi = cx.gains.view(-1, 1, 1) * pos.unsqueeze(0)
    return cx._grid_code(phi)


def view(pos, landmarks, sa=0.25):
    """Retinal PANORAMA at pos: for each bearing bin, brightness from landmarks at that bearing,
    nearer ones looming larger. pos (n,2), landmarks (M,2) -> (n, B_BINS)."""
    d = landmarks.unsqueeze(0) - pos.unsqueeze(1)                    # (n,M,2)
    dist = d.norm(dim=-1)                                            # (n,M)
    bearing = torch.atan2(d[..., 1], d[..., 0])                     # (n,M)
    bins = torch.linspace(-math.pi, math.pi, B_BINS + 1)[:-1]
    diff = bearing.unsqueeze(-1) - bins.view(1, 1, B_BINS)
    diff = torch.atan2(diff.sin(), diff.cos())                      # wrap to [-pi,pi]
    resp = torch.exp(-diff ** 2 / (2 * sa ** 2)) / (0.3 + dist).unsqueeze(-1)
    return resp.sum(1)                                              # (n, B_BINS)


def walks(n, T, R, seed, origin=False):
    g = torch.Generator().manual_seed(seed)
    # origin=True: start at home (phase 0) so integrated displacement == absolute position (path
    # integration). origin=False: random starts, to sample diverse views for the visual front-end.
    pos = torch.zeros(n, 2) if origin else (torch.rand(n, 2, generator=g) * 2 - 1) * R
    traj = [pos.clone()]
    for _ in range(T):
        h = torch.rand(n, generator=g) * 2 * math.pi; s = torch.rand(n, generator=g) * 0.6 + 0.2
        pos = (pos + torch.stack([s * h.cos(), s * h.sin()], -1)).clamp(-R, R)
        traj.append(pos.clone())
    return torch.stack(traj, 1)                                     # (n, T+1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--M", type=int, default=16)                    # landmarks in the world
    a = ap.parse_args()
    R = a.R
    torch.manual_seed(0)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    landmarks = (torch.rand(a.M, 2) * 2 - 1) * R * 1.3              # the visual world (fixed)

    # ---- (1) learn the visual front-end: optic flow (view_t, view_{t+1}) -> self-motion velocity ----
    eye = nn.Sequential(nn.Linear(2 * B_BINS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(eye.parameters(), lr=1e-3)
    for ep in range(400):
        tr = walks(256, 2, R, 1 + ep)
        v0 = view(tr[:, 0], landmarks); v1 = view(tr[:, 1], landmarks)
        vel = tr[:, 1] - tr[:, 0]                                   # true self-motion (efference copy = target)
        opt.zero_grad(); F.mse_loss(eye(torch.cat([v0, v1], -1)), vel).backward(); opt.step()
    with torch.no_grad():
        te = walks(3000, 2, R, 9999)
        vpred = eye(torch.cat([view(te[:, 0], landmarks), view(te[:, 1], landmarks)], -1))
        vtrue = te[:, 1] - te[:, 0]
        ego_err = (vpred - vtrue).norm(dim=-1).mean().item()
        ego_cos = F.cosine_similarity(vpred, vtrue, dim=-1).mean().item()

    # ---- position decoder for the grid map (trained on TRUE-velocity grid codes) ----
    def integrate(vel_seq):                                         # (n,T,2) -> final grid code
        v3d = torch.cat([vel_seq, torch.zeros(*vel_seq.shape[:2], 1)], -1)
        return cx(v3d, return_cells=True)[1]
    dec = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
    dopt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for ep in range(400):
        tr = walks(256, 10, R, 5000 + ep, origin=True)
        vel = tr[:, 1:] - tr[:, :-1]
        with torch.no_grad():
            gc = integrate(vel)
        dopt.zero_grad(); F.mse_loss(dec(gc), tr[:, -1]).backward(); dopt.step()

    # ---- (2) path-integrate the VISION-derived velocity through the grid map; localize ----
    @torch.no_grad()
    def pi_error(T, use_vision):
        tr = walks(2000, T, R, 20000 + T, origin=True)
        if use_vision:                                             # estimate each step's velocity from optic flow
            vest = []
            for t in range(T):
                vest.append(eye(torch.cat([view(tr[:, t], landmarks), view(tr[:, t + 1], landmarks)], -1)))
            vel = torch.stack(vest, 1)
        else:
            vel = tr[:, 1:] - tr[:, :-1]                           # true velocity (efference ceiling)
        return (dec(integrate(vel)) - tr[:, -1]).norm(dim=-1).mean().item()

    Ts = [6, 12, 18, 24]
    vis = {T: round(pi_error(T, True), 3) for T in Ts}
    tru = {T: round(pi_error(T, False), 3) for T in Ts}

    out = {"landmarks": a.M, "egomotion_error": round(ego_err, 4), "egomotion_cos": round(ego_cos, 3),
           "vision_pi_error_by_len": vis, "true_velocity_pi_error_by_len": tru}
    print("EMBODIMENT — the grid map grounded in vision (optic-flow self-motion):", flush=True)
    print(f"  visual self-motion estimate: error {ego_err:.3f} (per-step move ~0.5), "
          f"direction cosine {ego_cos:.3f}", flush=True)
    print(f"  path integration FROM VISION, position error by length: {vis}", flush=True)
    print(f"  vs true-velocity ceiling:                               {tru}", flush=True)
    print(f"  -> the agent localizes from what it SEES, not from hand-given motion", flush=True)

    # SVG: world + one true path vs the path reconstructed purely from vision
    @torch.no_grad()
    def vis_traj(seed):
        tr = walks(1, 26, R, seed, origin=True)[0]                              # (T+1,2)
        v = [eye(torch.cat([view(tr[t:t + 1], landmarks), view(tr[t + 1:t + 2], landmarks)], -1))[0] for t in range(26)]
        rec = torch.cumsum(torch.stack([torch.zeros(2)] + v), 0) + tr[0]   # visually dead-reckoned path
        return tr, rec
    svg_embodiment(landmarks, vis_traj(31), R, vis, tru, Ts, "results/embodiment.svg")
    os.makedirs("results", exist_ok=True)
    with open("results/embodiment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/embodiment.json and results/embodiment.svg", flush=True)


def svg_embodiment(landmarks, traj, R, vis, tru, Ts, out):
    true_path, rec_path = traj
    sz = 230; pad = 20; py = 56; qx = pad + sz + 70; qw = 250; qh = sz
    W = qx + qw + pad; H = sz + 90
    Rv = R * 1.35
    def X(x, off=pad): return off + (x + Rv) / (2 * Rv) * sz
    def Y(y): return py + (Rv - y) / (2 * Rv) * sz
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Embodiment: path integration from VISION (optic-flow self-motion)</text>')
    e.append(f'<rect x="{pad}" y="{py}" width="{sz}" height="{sz}" fill="#0b1324"/>')
    for L in landmarks:                                            # the visual world
        e.append(f'<circle cx="{X(L[0].item()):.1f}" cy="{Y(L[1].item()):.1f}" r="3" fill="#f6c945"/>')
    tp = " ".join(f"{X(p[0].item()):.1f},{Y(p[1].item()):.1f}" for p in true_path)
    rp = " ".join(f"{X(p[0].item()):.1f},{Y(p[1].item()):.1f}" for p in rec_path)
    e.append(f'<polyline points="{tp}" fill="none" stroke="#2ca25f" stroke-width="2.4"/>')        # true
    e.append(f'<polyline points="{rp}" fill="none" stroke="#e6550d" stroke-width="2.0" stroke-dasharray="5,3"/>')  # from vision
    e.append(f'<text x="{pad}" y="{py+sz+18}" font-size="10.5" fill="#5b6b8c">yellow = landmarks · '
             f'green = true path · orange dashed = path reconstructed from vision alone</text>')
    # right: PI error vs length, vision vs ceiling
    ymax = max(max(vis.values()), max(tru.values())) * 1.15 + 1e-6
    def QX(t): return qx + (t - Ts[0]) / (Ts[-1] - Ts[0]) * qw
    def QY(v): return py + qh - v / ymax * qh
    e.append(f'<text x="{qx}" y="50" font-size="11.5" fill="#28324a">position error vs path length</text>')
    e.append(f'<line x1="{qx}" y1="{py+qh}" x2="{qx+qw}" y2="{py+qh}" stroke="#33415c"/>'
             f'<line x1="{qx}" y1="{py}" x2="{qx}" y2="{py+qh}" stroke="#33415c"/>')
    for series, col in [(tru, "#2ca25f"), (vis, "#e6550d")]:
        pts = " ".join(f"{QX(t):.1f},{QY(series[t]):.1f}" for t in Ts)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for t in Ts:
            e.append(f'<circle cx="{QX(t):.1f}" cy="{QY(series[t]):.1f}" r="3" fill="{col}"/>')
    for t in Ts:
        e.append(f'<text x="{QX(t):.1f}" y="{py+qh+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">T={t}</text>')
    ly = py + 6
    for col, lab in [("#2ca25f", "true velocity (ceiling)"), ("#e6550d", "from vision")]:
        e.append(f'<rect x="{qx+qw-150}" y="{ly}" width="13" height="4" fill="{col}"/>')
        e.append(f'<text x="{qx+qw-133}" y="{ly+6}" font-size="10.5" fill="#28324a">{lab}</text>'); ly += 18
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
