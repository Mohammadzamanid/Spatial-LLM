"""
src/eval/space_time_circuit.py

CIRCUIT EMBEDDING — do SPACE and TIME share one circuit? In hippocampus, place cells, time cells, and
CONJUNCTIVE space x time cells coexist in a single population (integration/competition of space and time;
Neuron 2024; bat CA1, Nature Neurosci 2023). Here ONE recurrent substrate
(src/models/neuro/temporal_cortex.py) is fed self-motion velocity AND a start pulse, and trained to
report BOTH position and elapsed time. We then MEASURE, per unit, how much of its activity is explained
by POSITION vs ELAPSED TIME (eta^2, variance explained) and classify pure place / pure time / conjunctive
-- nothing imposed. A BOUNDED box keeps position ~decorrelated from elapsed time so the two tunings are
separable.

Multi-seed, mean +/- 95% CI. Writes results/space_time_circuit.json + .svg.

    python -m src.eval.space_time_circuit --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.temporal_cortex import TemporalCortex

T = 40; HIDDEN = 128; NOISE = 0.06; ACT_COST = 1e-3; LBOX = 1.0
GP = 6                 # position grid (GP x GP) for spatial tuning
TB = 10                # time bins for temporal tuning
ETA = 0.08             # eta^2 threshold for "tuned"


def walk(B, gen):
    v = torch.zeros(B, T, 2); pos = torch.zeros(B, T, 2)
    p = torch.zeros(B, 2); vel = torch.zeros(B, 2)
    for t in range(T):
        vel = 0.8 * vel + 0.2 * torch.randn(B, 2, generator=gen) * 0.5
        p = p + vel
        over = p.abs() > LBOX
        p = torch.where(over, torch.sign(p) * (2 * LBOX) - p, p)
        vel = torch.where(over, -vel, vel)
        v[:, t] = vel; pos[:, t] = p
    return v, pos


def eta2(a, lab, nbin):
    tot = a.var(unbiased=False) + 1e-9; bet = 0.0; m = a.mean()
    for k in range(nbin):
        sel = lab == k
        if sel.any():
            bet = bet + sel.float().mean() * (a[sel].mean() - m) ** 2
    return (bet / tot).item()


def run_seed(seed, iters=2000, want_arrays=False):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=HIDDEN, n_in=3)
    ph = nn.Linear(HIDDEN, 2); th = nn.Linear(HIDDEN, 1)
    opt = torch.optim.Adam(list(cx.parameters()) + list(ph.parameters()) + list(th.parameters()), 3e-3)
    for _ in range(iters):
        v, pos = walk(96, g); x = torch.zeros(96, T, 3); x[:, :, :2] = v; x[:, 0, 2] = 1.0
        R = cx.dynamics(x, noise=NOISE, gen=g)
        probe = torch.randint(T // 5, T, (96,), generator=g); rp = R[torch.arange(96), probe]
        loss = ((ph(rp) - pos[torch.arange(96), probe]) ** 2).mean() \
            + ((th(rp).squeeze(-1) - probe.float() / T) ** 2).mean() + ACT_COST * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        v, pos = walk(300, g); x = torch.zeros(300, T, 3); x[:, :, :2] = v; x[:, 0, 2] = 1.0
        R = cx.dynamics(x, noise=NOISE, gen=g)
        probe = torch.randint(T // 5, T, (300,), generator=g); rp = R[torch.arange(300), probe]
        pos_mae = (ph(rp) - pos[torch.arange(300), probe]).abs().mean().item()
        time_mae = (th(rp).squeeze(-1) - probe.float() / T).abs().mean().item() * T
        A = R.reshape(-1, HIDDEN); P = pos.reshape(-1, 2); tt = torch.arange(T).float().repeat(300)
        gx = ((P[:, 0] + LBOX) / (2 * LBOX) * GP).clamp(0, GP - 0.01).long()
        gy = ((P[:, 1] + LBOX) / (2 * LBOX) * GP).clamp(0, GP - 0.01).long()
        pbin = gx * GP + gy; tbin = (tt / T * TB).clamp(0, TB - 0.01).long()
        se = torch.zeros(HIDDEN); te = torch.zeros(HIDDEN); active = torch.zeros(HIDDEN, dtype=torch.bool)
        for u in range(HIDDEN):
            a = A[:, u]
            if a.std() < 1e-3:
                continue
            active[u] = True; se[u] = eta2(a, pbin, GP * GP); te[u] = eta2(a, tbin, TB)
        is_sp = (se > ETA) & active; is_tm = (te > ETA) & active
        n_act = int(active.sum())
        place = int((is_sp & ~is_tm).sum()); timec = int((is_tm & ~is_sp).sum())
        conj = int((is_sp & is_tm).sum())
        out = {"pos_mae": pos_mae, "time_mae": time_mae, "n_active": n_act,
               "frac_place": place / n_act, "frac_time": timec / n_act, "frac_conjunctive": conj / n_act}
    arr = None
    if want_arrays:
        # example cells for the figure: a pure place cell and a pure time cell
        pu = int((se * (is_sp & ~is_tm).float() - 1e3 * (~(is_sp & ~is_tm)).float()).argmax())
        tu = int((te * (is_tm & ~is_sp).float() - 1e3 * (~(is_tm & ~is_sp)).float()).argmax())
        pmap = torch.zeros(GP, GP)
        for i in range(GP):
            for j in range(GP):
                sel = (gx == i) & (gy == j)
                if sel.any():
                    pmap[i, j] = A[sel, pu].mean()
        ttune = torch.stack([A[tbin == k, tu].mean() if (tbin == k).any() else torch.tensor(0.) for k in range(TB)])
        arr = {"pmap": pmap, "ttune": ttune}
    return (out, arr) if want_arrays else (out, None)


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    per = []; arr0 = None
    for s in range(a.seeds):
        out, arr = run_seed(s, a.iters, want_arrays=(s == 0))
        if s == 0:
            arr0 = arr
        per.append(out)
        print(f"  seed {s}: pos-MAE {out['pos_mae']:.2f} time-MAE {out['time_mae']:.1f} | PLACE {out['frac_place']:.0%} "
              f"TIME {out['frac_time']:.0%} CONJUNCTIVE {out['frac_conjunctive']:.0%}", flush=True)
    keys = ["frac_place", "frac_time", "frac_conjunctive", "pos_mae", "time_mae"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    lab = {"frac_place": "PURE PLACE cells (space-tuned, time-invariant)",
           "frac_time": "PURE TIME cells (time-tuned, space-invariant)",
           "frac_conjunctive": "CONJUNCTIVE space x time cells (both)",
           "pos_mae": "decode POSITION, MAE (box half-width = 1.0)",
           "time_mae": "decode ELAPSED TIME, MAE steps (of %d)" % T}
    print(f"\nCIRCUIT EMBEDDING — space & time in one population (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 74, flush=True)
    for k in keys:
        print(f"  {lab[k]:58} {agg[k][0]:+.3f} ± {agg[k][1]:.3f}", flush=True)
    print(f"\n  -> ONE recurrent circuit, fed velocity + a start pulse and asked for position AND elapsed time, "
          f"develops PLACE ({agg['frac_place'][0]:.0%}), TIME ({agg['frac_time'][0]:.0%}), and CONJUNCTIVE "
          f"space x time ({agg['frac_conjunctive'][0]:.0%}) cells COEXISTING in one population (Neuron 2024) — "
          f"decoding position (MAE {agg['pos_mae'][0]:.2f}) and time (MAE {agg['time_mae'][0]:.1f}) together.", flush=True)
    out = {"n_seeds": a.seeds, "T": T, "hidden": HIDDEN, "iters": a.iters,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/space_time_circuit.json", "w"), indent=2)
    if arr0 is not None:
        svg(agg, arr0, "results/space_time_circuit.svg")
    print("\nwrote results/space_time_circuit.json and results/space_time_circuit.svg", flush=True)


def _cmap(v):
    v = max(0.0, min(1.0, float(v)))
    r = int(68 + v * (253 - 68)); gg = int(1 + v * (231 - 1)); b = int(84 + v * (37 - 84))
    return f"#{r:02x}{gg:02x}{b:02x}"


def svg(agg, arr, out):
    pmap = arr["pmap"]; ttune = arr["ttune"]
    pn = (pmap - pmap.min()) / (pmap.max() - pmap.min() + 1e-9)
    pad = 50; cell = 26; mapw = GP * cell; pw = 250; gap = 56
    W = pad + mapw + gap + pw + gap + 200 + pad; Hh = 80 + max(mapw, 150) + 40
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'One circuit: place, time &amp; conjunctive space&#215;time cells coexist</text>')
    oy = 56
    # place-cell rate map
    e.append(f'<text x="{pad}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">a PLACE cell (spatial rate map)</text>')
    for i in range(GP):
        for j in range(GP):
            e.append(f'<rect x="{pad+j*cell}" y="{oy+i*cell}" width="{cell+0.5}" height="{cell+0.5}" fill="{_cmap(pn[i,j].item())}"/>')
    # time-cell tuning curve
    tx = pad + mapw + gap
    e.append(f'<text x="{tx}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">a TIME cell (tuning vs elapsed time)</text>')
    tn = (ttune - ttune.min()) / (ttune.max() - ttune.min() + 1e-9); th_ = mapw
    e.append(f'<line x1="{tx}" y1="{oy+th_}" x2="{tx+pw}" y2="{oy+th_}" stroke="#33415c"/>')
    pts = " ".join(f"{tx + k/(TB-1)*pw:.1f},{oy+th_ - tn[k].item()*(th_-8):.1f}" for k in range(TB))
    e.append(f'<polyline points="{pts}" fill="none" stroke="#2ca25f" stroke-width="2.6"/>')
    e.append(f'<text x="{tx+pw/2:.0f}" y="{oy+th_+16:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">elapsed time &#8594;</text>')
    # fraction bars
    bx = tx + pw + gap; base = oy + mapw; bw = 44
    e.append(f'<text x="{bx}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">population</text>')
    for i, (k, lb, col) in enumerate([("frac_place", "place", "#3182bd"), ("frac_time", "time", "#2ca25f"),
                                      ("frac_conjunctive", "conj.", "#e6550d")]):
        val = agg[k][0]; h = val * mapw; xb = bx + i * (bw + 14)
        e.append(f'<rect x="{xb}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{xb+bw/2:.0f}" y="{base-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{val:.0%}</text>')
        e.append(f'<text x="{xb+bw/2:.0f}" y="{base+14:.0f}" font-size="9.5" fill="#28324a" text-anchor="middle">{lb}</text>')
    e.append(f'<line x1="{bx}" y1="{base}" x2="{bx+3*(bw+14)}" y2="{base}" stroke="#33415c"/>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
