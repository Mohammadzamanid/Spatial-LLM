"""
src/eval/stats.py

Statistical rigor pass: run the flagship cognitive-map results across MANY SEEDS and report
mean ± 95% CI (not single runs). This is the minimum bar any publication venue requires — it turns
"it worked once" into "it works, n seeds, with error bars". Each metric below re-implements the core
measurement of its eval script faithfully, in a seed loop.

    python -m src.eval.stats --seeds 8     # -> results/stats.json
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
    phi = cx.gains.view(-1, 1, 1) * pos.unsqueeze(0)
    return cx._grid_code(phi)


def cortex():
    return _HexGridModules(64, n_modules=6, base_spacing=1.6)


def rand_pos(n, R, g):
    return (torch.rand(n, 2, generator=g) * 2 - 1) * R


def winding(n, T, R, g, origin=False):
    pos = torch.zeros(n, 2) if origin else rand_pos(n, R, g)
    out = [pos.clone()]
    for _ in range(T):
        h = torch.rand(n, generator=g) * 2 * math.pi; s = torch.rand(n, generator=g) * 0.6 + 0.2
        pos = (pos + torch.stack([s * h.cos(), s * h.sin()], -1)).clamp(-R, R); out.append(pos.clone())
    return torch.stack(out, 1)


# ---------------------------------------------------------------- flagship metrics (per seed)
def planning_metric(seed, R=3.0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    cx = cortex()
    posT = rand_pos(8000, R, g); dec = nn.Sequential(nn.Linear(cx.K * cx.M, 256), nn.ReLU(), nn.Linear(256, 2))
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(500):
        opt.zero_grad(); F.mse_loss(dec(grid_code(cx, posT)), posT).backward(); opt.step()
    A = winding(3000, 14, R, g)[:, -1]; B = winding(3000, 14, R, g)[:, -1]
    with torch.no_grad():
        planned = dec(grid_code(cx, B)) - dec(grid_code(cx, A))
    cos = F.cosine_similarity(planned, B - A, dim=1).clamp(-1, 1)
    return {"shortcut_dir_error_deg": torch.rad2deg(torch.acos(cos)).mean().item(),
            "navigable_frac": (torch.rad2deg(torch.acos(cos)) < 15).float().mean().item()}


def relational_metric(seed, N=12, D=0.5, noise=0.8):
    torch.manual_seed(seed)
    cx = cortex(); ranks = torch.arange(N).float()
    item_pos = torch.stack([ranks * D - (N - 1) * D / 2, torch.zeros(N)], -1)
    codes = grid_code(cx, item_pos)
    C = nn.Sequential(nn.Linear(2 * cx.K * cx.M, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(C.parameters(), lr=1e-3)
    adj = torch.tensor([(i, i + 1) for i in range(N - 1)] + [(i + 1, i) for i in range(N - 1)])

    def feat(p):
        return torch.cat([codes[p[:, 0]], codes[p[:, 1]]], -1), (ranks[p[:, 0]] > ranks[p[:, 1]]).float()
    for _ in range(3000):
        x, y = feat(adj); opt.zero_grad(); F.binary_cross_entropy_with_logits(C(x).squeeze(-1), y).backward(); opt.step()
    nonadj = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) >= 2])

    @torch.no_grad()
    def acc(p, d, t=12):
        a, b = p[:, 0], p[:, 1]; yy = (ranks[a] > ranks[b]).float(); c = 0.0
        for _ in range(t):
            ci = codes[a] + d * torch.randn_like(codes[a]); cj = codes[b] + d * torch.randn_like(codes[b])
            c += ((C(torch.cat([ci, cj], -1)).squeeze(-1) > 0).float() == yy).float().mean().item()
        return c / t
    # symbolic distance effect: correlation of accuracy with rank-distance (should be +)
    dvals = list(range(1, N)); accs = [acc(torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) == dd]), noise, 8) for dd in dvals]
    dt = torch.tensor(dvals).float(); at = torch.tensor(accs)
    sde_corr = (F.cosine_similarity((dt - dt.mean()).unsqueeze(0), (at - at.mean()).unsqueeze(0))).item()
    return {"transitive_inference_acc": acc(nonadj, noise), "sde_distance_acc_correlation": sde_corr}


def continual_metric(seed, K=20, R=3.0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    cx = cortex(); locs = rand_pos(K, R, g)
    def nrm(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    W = torch.stack([nrm(grid_code(cx, locs[i:i + 1]))[0] for i in range(K)])
    clf = nn.Sequential(nn.Linear(cx.K * cx.M, 128), nn.ReLU(), nn.Linear(128, K))
    opt = torch.optim.Adam(clf.parameters(), lr=5e-3)
    for i in range(K):
        for _ in range(30):
            x = grid_code(cx, locs[i] + 0.15 * torch.randn(64, 2)); opt.zero_grad()
            F.cross_entropy(clf(x), torch.full((64,), i)).backward(); opt.step()

    @torch.no_grad()
    def recall(pred, t=16):
        a = torch.zeros(K)
        for _ in range(t):
            probe = grid_code(cx, locs + 0.15 * torch.randn(K, 2)); a += (pred(probe) == torch.arange(K)).float()
        return (a / t).mean().item()
    return {"hebbian_recall": recall(lambda c: (nrm(c) @ W.t()).argmax(1)),
            "gradient_recall": recall(lambda c: clf(c).argmax(1))}


def goal_metric(seed, R=3.0, rho=0.6, gamma=0.9, epochs=200):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    cx = cortex(); G = rand_pos(1, R * 0.6, g)[0]
    def rew(p): return (((p - G) ** 2).sum(-1) < rho ** 2).float()
    V = nn.Sequential(nn.Linear(cx.K * cx.M, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(V.parameters(), lr=1e-3)
    for ep in range(epochs):
        tr = winding(512, 40, R, torch.Generator().manual_seed(1000 * seed + ep))
        p0 = tr[:, :-1].reshape(-1, 2); p1 = tr[:, 1:].reshape(-1, 2); r0 = rew(p0)
        v0 = V(grid_code(cx, p0)).squeeze(-1)
        with torch.no_grad(): v1 = V(grid_code(cx, p1)).squeeze(-1)
        ((r0 + gamma * v1 * (1 - r0) - v0) ** 2).mean().backward(); opt.step(); opt.zero_grad()

    @torch.no_grad()
    def nav(policy, n=300, steps=60, st=0.4):
        pos = rand_pos(n, R, g); reached = torch.zeros(n, dtype=torch.bool)
        angs = torch.linspace(0, 2 * math.pi, 9)[:-1]
        for _ in range(steps):
            if policy == "value":
                cand = (pos.unsqueeze(1) + st * torch.stack([angs.cos(), angs.sin()], -1).unsqueeze(0)).clamp(-R, R)
                pos = cand[torch.arange(n), V(grid_code(cx, cand.reshape(-1, 2))).reshape(n, 8).argmax(1)]
            else:
                h = torch.rand(n) * 2 * math.pi; pos = (pos + st * torch.stack([h.cos(), h.sin()], -1)).clamp(-R, R)
            reached |= ((pos - G) ** 2).sum(-1) < rho ** 2
        return reached.float().mean().item()
    return {"value_nav_success": nav("value"), "random_nav_success": nav("random")}


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); m = t.mean().item(); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(m, 3), round(1.96 * sd / math.sqrt(n), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=8); a = ap.parse_args()
    seeds = list(range(a.seeds))
    suites = {"planning": planning_metric, "relational": relational_metric,
              "continual": continual_metric, "goal_navigation": goal_metric}
    out = {"n_seeds": a.seeds, "results": {}}
    print(f"multi-seed statistics (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 64, flush=True)
    for name, fn in suites.items():
        per = [fn(s) for s in seeds]
        agg = {}
        for k in per[0]:
            m, ci = ci95([p[k] for p in per]); agg[k] = {"mean": m, "ci95": ci}
        out["results"][name] = agg
        print(f"[{name}]", flush=True)
        for k, v in agg.items():
            print(f"    {k:32} {v['mean']:.3f} ± {v['ci95']:.3f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/stats.json", flush=True)


if __name__ == "__main__":
    main()
