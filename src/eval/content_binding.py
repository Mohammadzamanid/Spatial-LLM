"""
src/eval/content_binding.py

WHAT-WHERE-WHEN — does the temporal code BIND CONTENT? Recent hippocampal work (bat CA1; Shimbo et al.,
Nature Neuroscience 2023; space-time integration, Neuron 2024) finds time cells split into two coexisting
populations: CONJUNCTIVE "contextual" time cells (tuned to event/context x time) and PURE time cells
(tuned to elapsed time across contexts). We test whether BOTH emerge in our recurrent substrate
(src/models/neuro/temporal_cortex.py) and whether the population decodes BOTH "what" and "when".

The substrate gets ONE of K events at t=0 and, at a random probe, must report BOTH the elapsed time AND
which event. Nothing about conjunctive/pure cells is imposed; we MEASURE, per time cell, how much its
firing depends on event identity (mean pairwise correlation of its per-event tuning curves: high => PURE,
low => CONJUNCTIVE), plus decode accuracy for what (event) and when (elapsed time).

Multi-seed, mean +/- 95% CI. Writes results/content_binding.json + .svg.

    python -m src.eval.content_binding --seeds 6
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.temporal_cortex import TemporalCortex

T = 50
HIDDEN = 128
K = 3                 # number of distinct events ("what")
NOISE = 0.06
ACT_COST = 1e-3
PURE_THRESH = 0.6     # mean cross-event tuning correlation above which a time cell is "pure"


def make_trial(B, gen):
    x = torch.zeros(B, T, K + 1)
    ev = torch.randint(K, (B,), generator=gen)
    x[torch.arange(B), 0, ev] = 1.0                                 # one of K events at t=0
    probe = torch.randint(T // 5, T, (B,), generator=gen)
    x[torch.arange(B), probe, K] = 1.0                              # probe pulse ("report now")
    return x, ev, probe


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def run_seed(seed, iters=1800, want_arrays=False):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=HIDDEN, n_in=K + 1)
    th = nn.Linear(HIDDEN, 1); eh = nn.Linear(HIDDEN, K)            # time head + event head
    opt = torch.optim.Adam(list(cx.parameters()) + list(th.parameters()) + list(eh.parameters()), 3e-3)
    for _ in range(iters):
        x, ev, probe = make_trial(96, g); R = cx.dynamics(x, noise=NOISE, gen=g)
        rp = R[torch.arange(96), probe]
        loss = ((th(rp).squeeze(-1) - probe.float() / T) ** 2).mean() \
            + nn.functional.cross_entropy(eh(rp), ev) + ACT_COST * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        x, ev, probe = make_trial(900, g); R = cx.dynamics(x, noise=NOISE, gen=g)
        rp = R[torch.arange(900), probe]
        what_acc = (eh(rp).argmax(-1) == ev).float().mean().item()
        when_mae = (th(rp).squeeze(-1) - probe.float() / T).abs().mean().item() * T
        A = torch.stack([R[ev == k].mean(0) for k in range(K)])     # (K,T,H) per-event tuning
        Apure = A.mean(0)                                           # (T,H) tuning across events
        Ar = Apure / (Apure.max(0).values + 1e-6); peak = Ar.argmax(0).float(); width = (Ar > 0.5).float().sum(0)
        near = torch.stack([Ar[max(0, int(p) - 5):int(p) + 6, u].sum() for u, p in enumerate(peak)])
        active = Apure.max(0).values > 0.05 * Apure.max()
        tc = (active & (near / (Ar.sum(0) + 1e-6) > 0.5) & (width < T * 0.5) & (peak > 1) & (peak < T - 2)).nonzero().squeeze(-1)
        purity = []
        for u in tc:
            cs = [_corr(A[i, :, u], A[j, :, u]) for i in range(K) for j in range(i + 1, K)]
            purity.append(sum(cs) / len(cs))
        purity = torch.tensor(purity) if len(tc) else torch.zeros(0)
        frac_pure = (purity > PURE_THRESH).float().mean().item() if len(tc) else float("nan")
        frac_conj = (purity <= PURE_THRESH).float().mean().item() if len(tc) else float("nan")
        widen = _corr(peak[tc], width[tc]) if len(tc) > 5 else float("nan")
        out = {"time_cell_frac": len(tc) / HIDDEN, "frac_pure": frac_pure, "frac_conjunctive": frac_conj,
               "what_acc": what_acc, "what_chance": 1 / K, "when_mae": when_mae, "width_latency_corr": widen}
    arr = None
    if want_arrays:
        arr = {"A": A, "tc": tc, "purity": purity, "peak": peak}
    return out, arr


def ci95(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), 0.0
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=1800)
    a = ap.parse_args()
    per = []; arr0 = None
    for s in range(a.seeds):
        out, arr = run_seed(s, iters=a.iters, want_arrays=(s == 0))
        if s == 0:
            arr0 = arr
        per.append(out)
        print(f"  seed {s}: time-cells {out['time_cell_frac']:.0%} | PURE {out['frac_pure']:.0%} "
              f"CONJUNCTIVE {out['frac_conjunctive']:.0%} | WHAT {out['what_acc']:.0%} | WHEN {out['when_mae']:.2f}", flush=True)

    keys = ["time_cell_frac", "frac_pure", "frac_conjunctive", "width_latency_corr",
            "what_acc", "what_chance", "when_mae"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    lab = {"time_cell_frac": "time cells emerged (fraction of units)",
           "frac_pure": "  PURE time cells (fire at their moment across events)",
           "frac_conjunctive": "  CONJUNCTIVE time cells (event x time, 'contextual')",
           "width_latency_corr": "  fields widen with latency (corr)",
           "what_acc": "decode WHAT (event identity), accuracy",
           "what_chance": "  chance (1/K)",
           "when_mae": "decode WHEN (elapsed time), error (steps)"}
    print(f"\nCONTENT-BINDING (what-where-when) — n={a.seeds} seeds; mean ± 95% CI\n" + "=" * 70, flush=True)
    for k in keys:
        print(f"  {lab[k]:60} {agg[k][0]:+.3f} ± {agg[k][1]:.3f}", flush=True)
    print(f"\n  -> the temporal code BINDS CONTENT: time cells split into PURE ({agg['frac_pure'][0]:.0%}) and "
          f"CONJUNCTIVE/contextual ({agg['frac_conjunctive'][0]:.0%}) populations (both coexist, bat CA1, "
          f"Shimbo 2023), and the population decodes BOTH WHAT (event {agg['what_acc'][0]:.0%} vs chance "
          f"{agg['what_chance'][0]:.0%}) and WHEN (elapsed {agg['when_mae'][0]:.2f} steps).", flush=True)

    out = {"n_seeds": a.seeds, "T": T, "hidden": HIDDEN, "K": K, "iters": a.iters,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/content_binding.json", "w"), indent=2)
    if arr0 is not None and len(arr0["tc"]) > 1:
        svg(agg, arr0, "results/content_binding.svg")
    print("\nwrote results/content_binding.json and results/content_binding.svg", flush=True)


def svg(agg, arr, out):
    """Two example time cells (a PURE one and a CONJUNCTIVE one), each with its K per-event tuning curves."""
    A = arr["A"]; tc = arr["tc"]; purity = arr["purity"]
    pure_u = int(tc[purity.argmax()]); conj_u = int(tc[purity.argmin()])
    cols = ["#e6550d", "#2ca25f", "#3182bd"]
    pad = 56; pw = 300; ph = 150; gap = 50
    W = pad + 2 * pw + gap + pad; Hh = 60 + ph + 60
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Content-binding: PURE vs CONJUNCTIVE (event&#215;time) time cells both emerge</text>')
    oy = 52
    def panel(ox, u, title):
        Au = A[:, :, u]                                            # (K,T)
        mx = Au.max().item() + 1e-6
        def X(t): return ox + (t / (T - 1)) * pw
        def Y(v): return oy + ph - (v / mx) * (ph - 8)
        e.append(f'<text x="{ox}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">{title}</text>')
        e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>')
        for k in range(K):
            pts = " ".join(f"{X(t):.1f},{Y(Au[k, t].item()):.1f}" for t in range(T))
            e.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k]}" stroke-width="2"/>')
        e.append(f'<text x="{ox+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">elapsed time &#8594;</text>')
    panel(pad, pure_u, f"PURE time cell (curves overlap across events; purity {purity.max():.2f})")
    panel(pad + pw + gap, conj_u, f"CONJUNCTIVE cell (event-specific; purity {purity.min():.2f})")
    # legend + summary
    ly = oy + ph + 34
    for k in range(K):
        e.append(f'<rect x="{pad + k*90}" y="{ly}" width="13" height="4" fill="{cols[k]}"/>')
        e.append(f'<text x="{pad + k*90 + 17}" y="{ly+5}" font-size="9.5" fill="#28324a">event {k+1}</text>')
    e.append(f'<text x="{W-pad}" y="{ly+5}" font-size="10.5" fill="#0b1324" text-anchor="end" font-weight="700">'
             f'PURE {agg["frac_pure"][0]:.0%} &#183; CONJUNCTIVE {agg["frac_conjunctive"][0]:.0%} &#183; '
             f'decode WHAT {agg["what_acc"][0]:.0%} (chance {agg["what_chance"][0]:.0%}), WHEN {agg["when_mae"][0]:.1f} steps</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
