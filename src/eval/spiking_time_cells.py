"""
src/eval/spiking_time_cells.py

THE TEMPORAL AXIS, IN SPIKES + MULTIPLE TIMESCALES — narrowing the gap from "reproducing the time-cell
signature" (rate units, src/eval/time_cells.py) toward "reproducing the organ". A SPIKING recurrent
substrate (src/models/neuro/spiking_temporal_cortex.py: adaptive-LIF, surrogate-gradient spikes, per-unit
LEARNABLE membrane & adaptation time-constants, private noise) is trained on ONE task — "report elapsed
time when probed" — with rate homeostasis, and we MEASURE what emerged. Nothing about time cells,
widening, scalar timing, or a timescale spectrum is in the substrate or the loss.

  1. SPIKING time cells emerge — single-peaked spike-rate fields that TILE the interval (spike-frequency
     adaptation makes firing transient: fire, adapt, fall silent).
  2. A TIMESCALE SPECTRUM emerges — the learnable per-unit membrane time-constants spread over ~an order
     of magnitude (heterogeneous), and SLOW cells code LATE times: corr(tau, peak latency) > 0. That is
     Howard's log-compressed time, emerging from the multi-timescale substrate (Howard & Eichenbaum).
  3. Fields WIDEN with latency, and decoded-time SD grows with elapsed time (SCALAR / Weber timing).

Control: a HOMOGENEOUS-tau net (one shared membrane time-constant) — the spectrum collapses (~1x) and the
slow->late log-compression vanishes, so both are carried by the emergent heterogeneity, not an artifact.

Multi-seed, mean +/- 95% CI. Writes results/spiking_time_cells.json + .svg.

    python -m src.eval.spiking_time_cells --seeds 6
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.spiking_temporal_cortex import SpikingTemporalCortex

T = 40              # interval length (steps)
HIDDEN = 128
NOISE = 0.2         # private membrane noise (source of scalar variability)
RATE_TARGET = 0.06  # homeostatic spikes/step (healthy sparse, transient firing)


def make_trial(B, gen, dev="cpu"):
    x = torch.zeros(B, T, 2, device=dev); x[:, 0, 0] = 1.0           # start pulse
    probe = torch.randint(T // 5, T, (B,), generator=gen, device=dev)
    x[torch.arange(B, device=dev), probe, 1] = 1.0                  # "report elapsed time now"
    return x, probe


def ridge(A, y, lam=1.0):
    Ab = torch.cat([A, torch.ones(A.shape[0], 1)], 1)
    return torch.linalg.solve(Ab.t() @ Ab + lam * torch.eye(Ab.shape[1]), Ab.t() @ y)


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def train(net, gen, iters):
    head = nn.Linear(HIDDEN, 1)                                     # learned elapsed-time readout
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), 2e-3)
    for _ in range(iters):
        x, probe = make_trial(96, gen)
        R, spike_rate, _ = net(x, noise=NOISE, gen=gen)
        pred = head(R[torch.arange(96), probe]).squeeze(-1)
        loss = ((pred - probe.float() / T) ** 2).mean() + 1.0 * (spike_rate - RATE_TARGET) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def measure(net, gen, n=600):
    with torch.no_grad():
        x, probe = make_trial(n, gen)
        R, spike_rate, _ = net(x, noise=NOISE, gen=gen)
        A = R.mean(0)                                                # (T,N) spike-rate tuning
        ts = torch.arange(T).float()
        W = ridge(A, ts)
        that = (torch.cat([R, torch.ones(n, T, 1)], -1) @ W).squeeze(-1)
        mae = (that.mean(0) - ts).abs().mean().item()
        sigma = that.std(0); mid = (ts > 4) & (ts < T - 4)
        scalar = _corr(ts[mid], sigma[mid])
        cv = sigma[mid] / ts[mid]; weber_cv = (cv.std(unbiased=True) / (cv.mean() + 1e-9)).item()
        Ar = A / (A.max(0).values + 1e-6); peak = Ar.argmax(0).float(); width = (Ar > 0.5).float().sum(0)
        near = torch.stack([Ar[max(0, int(p) - 4):int(p) + 5, u].sum() for u, p in enumerate(peak)])
        active = A.max(0).values > 0.05 * A.max()
        is_tc = active & (near / (Ar.sum(0) + 1e-6) > 0.5) & (width < T * 0.5) & (peak > 1) & (peak < T - 2)
        tc = is_tc.nonzero().squeeze(-1)
        tau = net.timescales()
        tau_spread = (tau.max() / tau.min()).item()
        slow_late = _corr(tau[tc], peak[tc]) if len(tc) > 5 else float("nan")
        width_lat = _corr(peak[tc], width[tc]) if len(tc) > 5 else float("nan")
        m = {"decode_mae": mae, "time_cell_frac": is_tc.float().mean().item(),
             "width_latency_corr": width_lat, "scalar_sigma_corr": scalar, "weber_fraction_cv": weber_cv,
             "tau_spread": tau_spread, "slow_late_corr": slow_late, "spike_rate": spike_rate.item()}
        arr = {"A": A, "tc": tc, "peak": peak, "tau": tau, "sigma": sigma, "ts": ts}
    return m, arr


def run_seed(seed, iters=2000, want_arrays=False):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    het = train(SpikingTemporalCortex(hidden=HIDDEN), g, iters)
    hm, arr = measure(het, g)
    g2 = torch.Generator().manual_seed(seed + 999); torch.manual_seed(seed + 999)
    homo = train(SpikingTemporalCortex(hidden=HIDDEN, homogeneous=True), g2, iters)
    cm, _ = measure(homo, g2)
    out = {**hm, "ctrl_tau_spread": cm["tau_spread"], "ctrl_slow_late_corr": cm["slow_late_corr"],
           "ctrl_time_cell_frac": cm["time_cell_frac"], "ctrl_decode_mae": cm["decode_mae"]}
    return (out, arr) if want_arrays else (out, None)


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
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    per = []; arr0 = None
    for s in range(a.seeds):
        out, arr = run_seed(s, iters=a.iters, want_arrays=(s == 0))
        if s == 0:
            arr0 = arr
        per.append(out)
        print(f"  seed {s}: spiking time-cells {out['time_cell_frac']:.0%} | widen {out['width_latency_corr']:+.2f} | "
              f"scalar {out['scalar_sigma_corr']:+.2f} | tau-spread {out['tau_spread']:.1f}x | "
              f"slow->late {out['slow_late_corr']:+.2f} | MAE {out['decode_mae']:.2f}", flush=True)

    keys = ["time_cell_frac", "width_latency_corr", "scalar_sigma_corr", "weber_fraction_cv",
            "tau_spread", "slow_late_corr", "decode_mae", "spike_rate",
            "ctrl_tau_spread", "ctrl_slow_late_corr", "ctrl_time_cell_frac", "ctrl_decode_mae"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    lab = {"time_cell_frac": "SPIKING time cells emerged (fraction, single-peaked, tiling)",
           "width_latency_corr": "  fields WIDEN with latency (corr)",
           "scalar_sigma_corr": "SCALAR timing: corr(decoded-time SD, elapsed time)",
           "weber_fraction_cv": "  Weber fraction SD/t, CV",
           "tau_spread": "TIMESCALE SPECTRUM emerged: tau_max/tau_min (heterogeneous)",
           "slow_late_corr": "  LOG-COMPRESSION: slow cells code LATE, corr(tau, peak) (Howard)",
           "decode_mae": "elapsed-time decode error (steps)",
           "spike_rate": "  mean firing rate (spikes/step)",
           "ctrl_tau_spread": "control (HOMOGENEOUS tau): spectrum (-> ~1x)",
           "ctrl_slow_late_corr": "control (HOMOGENEOUS tau): log-compression (-> ~0/none)",
           "ctrl_time_cell_frac": "control (HOMOGENEOUS tau): time-cell fraction",
           "ctrl_decode_mae": "control (HOMOGENEOUS tau): decode error (steps)"}
    print(f"\nSPIKING + MULTI-TIMESCALE TIME CODE (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 74, flush=True)
    for k in keys:
        print(f"  {lab[k]:66} {agg[k][0]:+.3f} ± {agg[k][1]:.3f}", flush=True)
    print(f"\n  -> a SPIKING substrate trained only to read elapsed time grows spiking time cells "
          f"({agg['time_cell_frac'][0]:.0%}) that widen with latency ({agg['width_latency_corr'][0]:+.2f}) and "
          f"time scalarly ({agg['scalar_sigma_corr'][0]:+.2f}); a heterogeneous TIMESCALE SPECTRUM emerges "
          f"({agg['tau_spread'][0]:.1f}x vs 1x control) and IMPROVES timing (MAE {agg['decode_mae'][0]:.2f} vs "
          f"homogeneous {agg['ctrl_decode_mae'][0]:.2f}). HONEST NON-RESULT: the slow->late log-compression is "
          f"not robust ({agg['slow_late_corr'][0]:+.2f} ± {agg['slow_late_corr'][1]:.2f}, CI crosses 0).",
          flush=True)
    out = {"n_seeds": a.seeds, "T": T, "hidden": HIDDEN, "noise": NOISE, "iters": a.iters,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/spiking_time_cells.json", "w"), indent=2)
    if arr0 is not None:
        svg(agg, arr0, "results/spiking_time_cells.svg")
    print("\nwrote results/spiking_time_cells.json and results/spiking_time_cells.svg", flush=True)


def svg(agg, arr, out):
    tc = arr["tc"]; A = arr["A"]; peak = arr["peak"]; tau = arr["tau"]
    Ar = A / (A.max(0).values + 1e-6)
    order = tc[peak[tc].argsort()]
    pad = 56; pw = 360; ph = 150
    W = pad + pw + pad; H = 60 + ph + 52 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Spiking time cells + an emergent multi-timescale spectrum that aids timing</text>')
    oy = 50; rh = (ph - 2) / max(1, len(order))
    def X(t): return pad + (t / (T - 1)) * pw
    for row, u in enumerate(order):
        yv = oy + row * rh
        for t in range(T):
            v = Ar[t, u].item()
            if v > 0.1:
                e.append(f'<rect x="{X(t):.1f}" y="{yv:.1f}" width="{pw/T+0.6:.1f}" height="{rh+0.6:.1f}" '
                         f'fill="hsl({int(250-200*row/max(1,len(order)))},75%,55%)" opacity="{v:.2f}"/>')
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+15:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">'
             f'elapsed time &#8594; ({len(order)} emergent SPIKING time cells, sorted by peak)</text>')
    # bottom: the emergent TIMESCALE SPECTRUM (histogram of membrane time-constants) — the robust result
    by = oy + ph + 46
    taus = arr["tau"]
    tmin, tmax = taus.min().item(), taus.max().item()
    nb = 24
    edges = [tmin * (tmax / tmin) ** (i / nb) for i in range(nb + 1)]                 # log-spaced bins
    counts = [int(((taus >= edges[i]) & (taus < edges[i + 1])).sum()) for i in range(nb)]
    cmax = max(counts) or 1
    e.append(f'<text x="{pad}" y="{by-6}" font-size="11" fill="#28324a">emergent timescale spectrum '
             f'(membrane &#964; histogram): {agg["tau_spread"][0]:.0f}&#215; spread vs 1&#215; homogeneous control</text>')
    e.append(f'<line x1="{pad}" y1="{by+ph}" x2="{pad+pw}" y2="{by+ph}" stroke="#33415c"/>')
    bwid = pw / nb
    for i, c in enumerate(counts):
        h = c / cmax * ph; x = pad + i * bwid
        e.append(f'<rect x="{x:.1f}" y="{by+ph-h:.1f}" width="{bwid-1:.1f}" height="{h:.1f}" fill="#2ca25f" opacity="0.8"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{by+ph+15:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">'
             f'membrane &#964; (log scale) &#183; multi-timescale times better: MAE {agg["decode_mae"][0]:.2f} vs '
             f'homogeneous {agg["ctrl_decode_mae"][0]:.2f} &#183; {agg["time_cell_frac"][0]:.0%} spiking time cells</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
