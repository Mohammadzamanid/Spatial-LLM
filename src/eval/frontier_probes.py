"""
src/eval/frontier_probes.py

HUNTING A SHARP CLAIM — two regimes where a FIXED brain-faithful prior might beat a model that must
LEARN to integrate, even the tough NoPE+sum Transformer that tied the grid code with abundant data.

  1. SAMPLE EFFICIENCY. Train each representation's readout on only N distinct trajectories (to
     convergence), test on fresh paths at 3x the training length. The grid code is FIXED — only a
     small readout is learned — so the integration "operation" is built in; a Transformer/GRU must
     DISCOVER integration from data and should need far more trajectories. If the gap at small N is
     large and a fair learned model cannot close it, that is a real inductive-bias advantage.

  2. NOISE ROBUSTNESS. Inject per-step velocity noise (real path integration is noisy). Does the
     fixed code degrade more gracefully than a learned integrator?

Reps: grid (fixed), NoPE+sum Transformer (the toughest baseline), GRU, place, oracle. Multi-seed,
mean ± 95% CI. Distance exact-acc at T=24 (the 3x extrapolation regime). Honest: if the NoPE+sum
Transformer keeps pace, we report the tie and stop claiming uniqueness.
Writes results/frontier_probes.json + results/frontier_probes.svg.

    python -m src.eval.frontier_probes --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.extrapolation import make_batch, GRURep, PlaceRep, OracleRep, metrics
from src.eval.ablations import GridRepN
from src.eval.seq_baselines import SeqTransformer


def make_reps(train_cover):
    return {
        "grid (fixed prior)": GridRepN(n_modules=6),
        "NoPE+sum Transformer": SeqTransformer(pos="none", pool="sum"),
        "GRU": GRURep(),
        "place (fixed prior)": PlaceRep(train_cover=train_cover),
        "oracle": OracleRep(),
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


# ---------------------------------------------------------------- probe 1: sample efficiency
def sample_efficiency_seed(seed, Ns, train_lengths=(6, 8, 10, 12), test_T=24, R=3.0,
                           steps=800, bs=128, n_eval=4000):
    cgen = torch.Generator().manual_seed(30_000 + seed)
    cover = round(torch.cat([make_batch(8000, T, cgen)[1] for T in train_lengths]).abs().quantile(0.99).item(), 3)
    egen = torch.Generator().manual_seed(90_000 + seed)
    ev, edisp = make_batch(n_eval, test_T, egen)                    # fresh test paths (3x length)

    out = {}
    for N in Ns:
        pgen = torch.Generator().manual_seed(40_000 + seed + N)     # a FIXED pool of N trajectories
        per_len = max(1, N // len(train_lengths))
        pool = {T: make_batch(per_len, T, pgen) for T in train_lengths}
        for name, rep_fac in _facs(cover).items():
            torch.manual_seed(seed)
            model = rep_fac()
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
            sgen = torch.Generator().manual_seed(60_000 + seed + N)
            for _ in range(steps):
                T = train_lengths[torch.randint(len(train_lengths), (1,), generator=sgen).item()]
                vN, dN = pool[T]
                idx = torch.randint(vN.shape[0], (min(bs, vN.shape[0]),), generator=sgen)
                opt.zero_grad(); F.mse_loss(model(vN[idx], dN[idx]), dN[idx]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = metrics(model(ev, edisp), edisp)["distance_exact_acc"]
            out.setdefault(name, {})[N] = acc
    return out


# ---------------------------------------------------------------- probe 2: noise robustness
def noisy_batch(n, T, gen, sigma):
    """Observed velocity is noisy. Returns (noisy velocity seq, noisy integrated displacement, CLEAN
    displacement target). EVERY code integrates the SAME noisy velocity (grid/place via the noisy
    integrated displacement; sequence models via the noisy sequence); the clean displacement is only
    the regression target — so no code is handed the answer."""
    h = torch.rand(n, T, generator=gen) * 2 * math.pi
    s = torch.rand(n, T, generator=gen) * 0.6 + 0.2
    v = torch.stack([s * h.cos(), s * h.sin()], -1)
    vobs = v + sigma * torch.randn(n, T, 2, generator=gen)
    return vobs, vobs.sum(1), v.sum(1)                              # (noisy seq, noisy disp, clean target)


def noise_seed(seed, sigmas, train_lengths=(6, 8, 10, 12), test_T=16, R=3.0, steps=800, bs=256, n_eval=4000):
    cgen = torch.Generator().manual_seed(30_000 + seed)
    cover = round(torch.cat([make_batch(8000, T, cgen)[1] for T in train_lengths]).abs().quantile(0.99).item(), 3)
    out = {}
    for sigma in sigmas:
        egen = torch.Generator().manual_seed(95_000 + seed)
        ev, edisp_obs, etgt = noisy_batch(n_eval, test_T, egen, sigma)
        for name, rep_fac in _facs(cover).items():
            torch.manual_seed(seed)
            model = rep_fac()
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
            tgen = torch.Generator().manual_seed(70_000 + seed)
            for _ in range(steps):
                T = train_lengths[torch.randint(len(train_lengths), (1,), generator=tgen).item()]
                vN, dN_obs, tgt = noisy_batch(bs, T, tgen, sigma)
                opt.zero_grad(); F.mse_loss(model(vN, dN_obs), tgt).backward(); opt.step()  # input noisy, target clean
            model.eval()
            with torch.no_grad():
                acc = metrics(model(ev, edisp_obs), etgt)["distance_exact_acc"]
            out.setdefault(name, {})[sigma] = acc
    return out


def _facs(cover):
    return {
        "grid (fixed prior)": lambda: GridRepN(n_modules=6),
        "NoPE+sum Transformer": lambda: SeqTransformer(pos="none", pool="sum"),
        "GRU": lambda: GRURep(),
        "place (fixed prior)": lambda: PlaceRep(train_cover=cover),
        "oracle": lambda: OracleRep(),
    }


def aggregate(per_seed, xs):
    names = list(per_seed[0].keys())
    return {nm: {x: dict(zip(("mean", "ci95"), ci95([s[nm][x] for s in per_seed]))) for x in xs} for nm in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    Ns = [16, 32, 64, 128, 512, 4096]
    sigmas = [0.0, 0.1, 0.2, 0.4]
    print(f"FRONTIER PROBES (n={a.seeds} seeds; mean ± 95% CI) — distance exact-acc\n" + "=" * 70, flush=True)

    se = aggregate([sample_efficiency_seed(s, Ns) for s in seeds], Ns)
    print("\n[1] SAMPLE EFFICIENCY — acc at T=24 (3x) vs # distinct training trajectories N:", flush=True)
    print("    " + "rep".ljust(22) + "".join(f"N={n}".rjust(11) for n in Ns), flush=True)
    for nm in se:
        print("    " + nm.ljust(22) + "".join(f"{se[nm][n]['mean']:.0%}".rjust(11) for n in Ns), flush=True)

    nz = aggregate([noise_seed(s, sigmas) for s in seeds], sigmas)
    print("\n[2] NOISE ROBUSTNESS — acc at T=16 vs per-step velocity noise sigma:", flush=True)
    print("    " + "rep".ljust(22) + "".join(f"s={sg}".rjust(11) for sg in sigmas), flush=True)
    for nm in nz:
        print("    " + nm.ljust(22) + "".join(f"{nz[nm][sg]['mean']:.0%}".rjust(11) for sg in sigmas), flush=True)

    out = {"n_seeds": a.seeds, "Ns": Ns, "sigmas": sigmas, "sample_efficiency": se, "noise": nz}
    os.makedirs("results", exist_ok=True)
    with open("results/frontier_probes.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_frontier(se, Ns, nz, sigmas, "results/frontier_probes.svg")
    print("\nwrote results/frontier_probes.json and results/frontier_probes.svg", flush=True)


PALETTE = {"grid (fixed prior)": "#e6550d", "NoPE+sum Transformer": "#21908c", "GRU": "#756bb1",
           "place (fixed prior)": "#3b528b", "oracle": "#9aa5b8"}


def _panel(e, agg, xs, ox, oy, pw, ph, title, xlabel, logx):
    def X(x):
        if logx:
            lo, hi = math.log(xs[0]), math.log(xs[-1]); return ox + (math.log(x) - lo) / (hi - lo) * pw
        return ox + xs.index(x) / (len(xs) - 1) * pw
    def Y(v): return oy + ph - v * ph
    e.append(f'<text x="{ox}" y="{oy-9}" font-size="12" font-weight="700" fill="#0b1324">{title}</text>')
    e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<line x1="{ox}" y1="{Y(vv):.1f}" x2="{ox+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{ox-6}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for x in xs:
        e.append(f'<text x="{X(x):.1f}" y="{oy+ph+13:.1f}" font-size="8.5" fill="#5b6b8c" text-anchor="middle">{x}</text>')
    e.append(f'<text x="{ox+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="9.5" fill="#28324a" text-anchor="middle">{xlabel}</text>')
    for nm, col in PALETTE.items():
        if nm not in agg:
            continue
        bt = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']+agg[nm][x]['ci95']):.1f}" for x in xs)
        bb = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']-agg[nm][x]['ci95']):.1f}" for x in reversed(xs))
        e.append(f'<polygon points="{bt} {bb}" fill="{col}" opacity="0.12"/>')
        pts = " ".join(f"{X(x):.1f},{Y(agg[nm][x]['mean']):.1f}" for x in xs)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for x in xs:
            e.append(f'<circle cx="{X(x):.1f}" cy="{Y(agg[nm][x]["mean"]):.1f}" r="2.6" fill="{col}"/>')


def svg_frontier(se, Ns, nz, sigmas, out):
    pad = 52; pw = 320; ph = 210; gap = 96
    W = pad + pw + gap + pw + pad; H = 64 + ph + 64
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Frontier probes: does the fixed prior win where a learned model must discover integration?</text>')
    e.append('<text x="26" y="43" font-size="10.5" fill="#5b6b8c">distance exact-acc &#183; mean &#177; 95% CI '
             '&#183; the NoPE+sum Transformer is the toughest fair baseline</text>')
    _panel(e, se, Ns, pad, 64, pw, ph, "A &#183; sample efficiency (acc @ T=24)", "# distinct training trajectories", True)
    _panel(e, nz, sigmas, pad + pw + gap, 64, pw, ph, "B &#183; noise robustness (acc @ T=16)", "per-step velocity noise sigma", False)
    ly = 64 + 6
    for nm, col in PALETTE.items():
        e.append(f'<rect x="{pad+pw+gap+pw-150}" y="{ly}" width="12" height="4" fill="{col}"/>')
        e.append(f'<text x="{pad+pw+gap+pw-135}" y="{ly+5}" font-size="8.5" fill="#28324a">{nm}</text>'); ly += 12
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
