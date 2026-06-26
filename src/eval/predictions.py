"""
src/eval/predictions.py

THE PLATFORM AS A HYPOTHESIS GENERATOR. Once the architecture reproduces the neuroscience by EMERGENCE,
it can be PERTURBED to make falsifiable predictions for experiment. Each run below sweeps one controlled
variable and records the model's quantitative consequence as a testable biological hypothesis. (We have
already run one full predict->test->falsify cycle: the model predicted slow-cells-code-late
log-compression, which did NOT replicate at n=6 — see results/spiking_time_cells.json.)

  P1. CONTENT LOAD -> CONJUNCTIVE FRACTION. Vary the number of distinct events K the temporal code must
      bind. Prediction: the fraction of CONJUNCTIVE (event x time) cells, vs PURE time cells, rises with
      content load. Biological test: in a timing task with more distinct cues, a larger share of time
      cells should be cue-selective (contextual), fewer purely temporal.

  P2. SPATIAL-INPUT RELIABILITY -> SPACE/TIME CELL MIX. Corrupt the self-motion (velocity) input in the
      one-circuit space+time model. Prediction: as spatial input degrades, the population REALLOCATES
      from place/conjunctive toward PURE TIME coding. Biological test: degrading vestibular/optic-flow
      input should shift the hippocampal mix toward time cells.

Multi-seed per condition. Writes results/predictions.json + .svg.

    python -m src.eval.predictions --seeds 3
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.temporal_cortex import TemporalCortex

T = 40; HIDDEN = 128; NOISE = 0.06; ACT = 1e-3; LBOX = 1.0


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def _time_cells(A):
    Ar = A / (A.max(0).values + 1e-6); peak = Ar.argmax(0).float(); width = (Ar > 0.5).float().sum(0)
    near = torch.stack([Ar[max(0, int(p) - 5):int(p) + 6, u].sum() for u, p in enumerate(peak)])
    act = A.max(0).values > 0.05 * A.max()
    return (act & (near / (Ar.sum(0) + 1e-6) > 0.5) & (width < T * 0.5) & (peak > 1) & (peak < T - 2)).nonzero().squeeze(-1)


# ----------------------------------------------------------- P1: content load -> conjunctive fraction
def p1_seed(K, seed, iters=1200):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=HIDDEN, n_in=K + 1)
    th = nn.Linear(HIDDEN, 1); eh = nn.Linear(HIDDEN, max(K, 1))
    opt = torch.optim.Adam(list(cx.parameters()) + list(th.parameters()) + list(eh.parameters()), 3e-3)
    def make(B):
        x = torch.zeros(B, T, K + 1); ev = torch.randint(K, (B,), generator=g)
        x[torch.arange(B), 0, ev] = 1.0
        probe = torch.randint(T // 5, T, (B,), generator=g); x[torch.arange(B), probe, K] = 1.0
        return x, ev, probe
    for _ in range(iters):
        x, ev, probe = make(96); R = cx.dynamics(x, noise=NOISE, gen=g); rp = R[torch.arange(96), probe]
        loss = ((th(rp).squeeze(-1) - probe.float() / T) ** 2).mean() + ACT * R.pow(2).mean()
        if K > 1:
            loss = loss + nn.functional.cross_entropy(eh(rp), ev)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x, ev, probe = make(600); R = cx.dynamics(x, noise=NOISE, gen=g)
        Aev = torch.stack([R[ev == k].mean(0) for k in range(K)])           # (K,T,H)
        A = Aev.mean(0); tc = _time_cells(A)
        if K == 1 or len(tc) < 3:
            return 0.0 if len(tc) else float("nan")                          # content-free -> no conjunctive
        conj = 0
        for u in tc:
            cs = [_corr(Aev[i, :, u], Aev[j, :, u]) for i in range(K) for j in range(i + 1, K)]
            if sum(cs) / len(cs) <= 0.6:                                     # low cross-event corr = conjunctive
                conj += 1
        return conj / len(tc)


# ----------------------------------------------------------- P2: spatial reliability -> space/time mix
def _walk(B, gen):
    v = torch.zeros(B, T, 2); pos = torch.zeros(B, T, 2); p = torch.zeros(B, 2); vel = torch.zeros(B, 2)
    for t in range(T):
        vel = 0.8 * vel + 0.2 * torch.randn(B, 2, generator=gen) * 0.5
        p = p + vel; over = p.abs() > LBOX
        p = torch.where(over, torch.sign(p) * 2 * LBOX - p, p); vel = torch.where(over, -vel, vel)
        v[:, t] = vel; pos[:, t] = p
    return v, pos


def _eta2(a, lab, nb):
    tot = a.var(unbiased=False) + 1e-9; bet = 0.0; m = a.mean()
    for k in range(nb):
        sel = lab == k
        if sel.any():
            bet = bet + sel.float().mean() * (a[sel].mean() - m) ** 2
    return (bet / tot).item()


def p2_seed(vnoise, seed, iters=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=HIDDEN, n_in=3); ph = nn.Linear(HIDDEN, 2); th = nn.Linear(HIDDEN, 1)
    opt = torch.optim.Adam(list(cx.parameters()) + list(ph.parameters()) + list(th.parameters()), 3e-3)
    for _ in range(iters):
        v, pos = _walk(96, g)
        vin = v + vnoise * torch.randn(v.shape, generator=g)                # corrupt the spatial input
        x = torch.zeros(96, T, 3); x[:, :, :2] = vin; x[:, 0, 2] = 1.0
        R = cx.dynamics(x, noise=NOISE, gen=g); probe = torch.randint(T // 5, T, (96,), generator=g)
        rp = R[torch.arange(96), probe]
        loss = ((ph(rp) - pos[torch.arange(96), probe]) ** 2).mean() \
            + ((th(rp).squeeze(-1) - probe.float() / T) ** 2).mean() + ACT * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        v, pos = _walk(300, g); vin = v + vnoise * torch.randn(v.shape, generator=g)
        x = torch.zeros(300, T, 3); x[:, :, :2] = vin; x[:, 0, 2] = 1.0
        R = cx.dynamics(x, noise=NOISE, gen=g); A = R.reshape(-1, HIDDEN)
        P = pos.reshape(-1, 2); tt = torch.arange(T).float().repeat(300)
        gx = ((P[:, 0] + LBOX) / (2 * LBOX) * 6).clamp(0, 5.99).long(); gy = ((P[:, 1] + LBOX) / (2 * LBOX) * 6).clamp(0, 5.99).long()
        pbin = gx * 6 + gy; tbin = (tt / T * 10).clamp(0, 9.99).long()
        place_or_conj = puretime = 0
        for u in range(HIDDEN):
            a = A[:, u]
            if a.std() < 1e-3:
                continue
            se = _eta2(a, pbin, 36); te = _eta2(a, tbin, 10)
            if se > 0.08:
                place_or_conj += 1                                          # spatially tuned (place or conjunctive)
            elif te > 0.08:
                puretime += 1
        tot = place_or_conj + puretime
        return puretime / tot if tot else float("nan")                      # PURE-TIME share of tuned cells


def ci(vals):
    vals = [v for v in vals if v == v]
    t = torch.tensor(vals); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 3), round(1.96 * sd / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    Ks = [1, 3, 5]; noises = [0.0, 0.6, 1.2]
    p1 = {K: ci([p1_seed(K, s) for s in range(a.seeds)]) for K in Ks}
    p2 = {nz: ci([p2_seed(nz, s) for s in range(a.seeds)]) for nz in noises}

    print("\n================ PLATFORM-GENERATED, FALSIFIABLE PREDICTIONS ================", flush=True)
    print("P1  content load K -> CONJUNCTIVE (event x time) fraction among time cells:", flush=True)
    for K in Ks:
        print(f"      K={K}: {p1[K][0]:.0%} ± {p1[K][1]:.0%}", flush=True)
    print(f"    PREDICTION: conjunctive fraction rises with content load "
          f"({p1[Ks[0]][0]:.0%} -> {p1[Ks[-1]][0]:.0%}). Biological test: more distinct cues in a timing\n"
          f"    task -> a larger share of time cells become cue-selective (contextual), fewer purely temporal.", flush=True)
    print("\nP2  spatial-input noise -> PURE-TIME share of tuned cells:", flush=True)
    for nz in noises:
        print(f"      vel-noise={nz}: {p2[nz][0]:.0%} ± {p2[nz][1]:.0%}", flush=True)
    print(f"    PREDICTION: degrading spatial input reallocates the population toward pure-time coding "
          f"({p2[noises[0]][0]:.0%} -> {p2[noises[-1]][0]:.0%}). Biological test: impairing vestibular/optic-flow\n"
          f"    input should shift the hippocampal mix away from place/conjunctive toward time cells.", flush=True)

    out = {"n_seeds": a.seeds,
           "P1_content_to_conjunctive": {str(K): {"mean": p1[K][0], "ci95": p1[K][1]} for K in Ks},
           "P2_spatialnoise_to_puretime": {str(nz): {"mean": p2[nz][0], "ci95": p2[nz][1]} for nz in noises}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/predictions.json", "w"), indent=2)
    svg(p1, Ks, p2, noises, "results/predictions.svg")
    print("\nwrote results/predictions.json and results/predictions.svg", flush=True)


def svg(p1, Ks, p2, noises, out):
    pad = 56; pw = 280; ph = 170; gap = 70; W = pad + 2 * pw + gap + pad; H = 70 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Platform-generated, falsifiable predictions</text>')
    oy = 56
    def trend(ox, xs, dat, title, xlab, col):
        e.append(f'<text x="{ox}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">{title}</text>')
        e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
                 f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
        for vv in (0.0, 0.5, 1.0):
            yy = oy + ph - vv * ph
            e.append(f'<text x="{ox-6}" y="{yy+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
        n = len(xs)
        def X(i): return ox + (i / (n - 1)) * pw
        def Y(v): return oy + ph - v * ph
        pts = " ".join(f"{X(i):.1f},{Y(dat[k][0]):.1f}" for i, k in enumerate(xs))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for i, k in enumerate(xs):
            m, c = dat[k]
            e.append(f'<line x1="{X(i):.1f}" y1="{Y(min(1,m+c)):.1f}" x2="{X(i):.1f}" y2="{Y(max(0,m-c)):.1f}" stroke="{col}"/>')
            e.append(f'<circle cx="{X(i):.1f}" cy="{Y(m):.1f}" r="3.5" fill="{col}"/>')
            e.append(f'<text x="{X(i):.1f}" y="{oy+ph+15:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">{k}</text>')
        e.append(f'<text x="{ox+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="10" fill="#28324a" text-anchor="middle">{xlab}</text>')
    trend(pad, Ks, p1, "P1: conjunctive fraction rises with content", "content load K", "#e6550d")
    trend(pad + pw + gap, noises, p2, "P2: pure-time share rises as space degrades", "velocity-input noise", "#2ca25f")
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
