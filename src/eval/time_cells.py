"""
src/eval/time_cells.py

THE TEMPORAL AXIS — hippocampal TIME CELLS (Eichenbaum 2014; MacDonald et al. 2011; Howard's
scale-invariant timing; Mau et al. 2018; Tacikowski et al. 2024), the dimension our purely-spatial
cortex omits.

Time cells fire at successive moments within a structured interval, tiling ELAPSED TIME the way place
cells tile space. Their defining, falsifiable signature is SCALAR (Weber) timing: temporal precision
degrades in proportion to elapsed time, because time fields WIDEN with their latency. We build a
faithful time-cell basis and verify, readout-independently where it matters, that:

  1. SCALAR TIMING / WEBER'S LAW (the signature, measured from the code geometry, not a decoder):
     the just-noticeable-difference in elapsed time JND(t) = 1 / ||da/dt|| (the inverse of the
     population's local discriminability) grows ~linearly with elapsed time when fields widen
     (corr(JND, t) -> 1), and is FLAT for fixed-width fields (corr -> 0). The Weber fraction JND/t
     stabilizes to a constant in the scale-invariant regime (a positive floor at short intervals is
     the generalized Weber law). The widening *causes* scalar timing; because JND is read off the
     code geometry itself it cannot be a trained-readout artifact.
  2. The code IDENTIFIES elapsed time and EVENT ORDER for separated events. We decode with the
     standard, parameter-free POPULATION VECTOR (center of mass) -- no fitting, nothing to tune, so
     "the code is usable" is not a property of a learned readout. (The widening late fields are
     deliberately collinear, which is exactly why a naive least-squares readout is ill-conditioned;
     the population vector sidesteps that and is the textbook neuroscience decode.)

Tiling is mildly jittered per seed (real time-cell centers are irregular) for an honest multi-seed
CI; it is held ~uniform in density on purpose so that the fixed-width control is genuinely flat and
WIDENING is the sole manipulated cause of scalar timing.

Multi-seed, mean +/- 95% CI. Writes results/time_cells.json + .svg.

    python -m src.eval.time_cells --seeds 8
"""
import argparse
import json
import math
import os

import torch

TMAX = 20.0            # length of the interval (elapsed-time units)
K = 50                 # number of time cells
WEBER = 0.20           # field-widening slope (Weber coefficient)
SIGMA0 = 0.4           # field width at t=0 (the short-interval timing floor)


def fields(centers, weber=WEBER, sigma0=SIGMA0, fixed=False):
    """Field widths. Scalar (Weber) widening sigma_k = sigma0*(1 + weber*center_k): later fields are
    broader, the empirical hallmark of time cells. fixed=True holds width at the mean (the control)."""
    if fixed:
        return torch.full_like(centers, sigma0 * (1 + weber * TMAX / 2))
    return sigma0 * (1 + weber * centers)


def population(t, centers, sig):
    return torch.exp(-((t.unsqueeze(-1) - centers.unsqueeze(0)) ** 2) / (2 * sig.unsqueeze(0) ** 2))


def com_decode(A, centers):
    """Population-vector / center-of-mass decode of elapsed time. Parameter-free, no fitting -- the
    standard neuroscience population decode, robust to the collinear widening fields."""
    return (A @ centers) / (A.sum(1) + 1e-9)


def jnd_vs_time(centers, fixed, dt=0.05):
    """Just-noticeable-difference in elapsed time JND(t) = 1 / ||da/dt||  (inverse local
    discriminability of the population code). Returns the interior t grid and JND(t), measured from
    the code geometry -- no decoder, so it cannot be a readout artifact."""
    sig = fields(centers, fixed=fixed)
    t = torch.linspace(1.0, TMAX - 1.0, 60)                       # interior (avoid edge truncation)
    speed = (population(t + dt, centers, sig) - population(t, centers, sig)).norm(dim=1) / dt
    return t, 1.0 / (speed + 1e-9)


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def run_seed(seed, n=4000):
    g = torch.Generator().manual_seed(seed)
    # mildly jittered tiling (real time-cell centers are irregular); density held ~uniform so the
    # fixed-width control stays flat and WIDENING is the only manipulated cause of scalar timing.
    base = torch.linspace(0.0, TMAX, K); spacing = TMAX / (K - 1)
    centers = (base + 0.35 * spacing * torch.randn(K, generator=g)).clamp(0.0, TMAX).sort().values

    # 1. scalar timing (Weber): JND grows ~linearly with elapsed time for widening fields, flat for fixed
    t, jnd_w = jnd_vs_time(centers, fixed=False)
    _, jnd_f = jnd_vs_time(centers, fixed=True)
    weber_corr_widening = _corr(t, jnd_w)
    weber_corr_fixed = _corr(t, jnd_f)
    # Weber fraction JND/t stabilizes to a constant in the scale-invariant regime (t > TMAX/2);
    # low coefficient of variation there = scale-invariant timing (a floor remains at short t).
    wf = (jnd_w / t)[t > TMAX / 2]
    weber_fraction_cv = (wf.std(unbiased=True) / (wf.mean() + 1e-9)).item()

    # 2. the code identifies elapsed time (parameter-free population-vector decode) and event order
    sig = fields(centers)
    te = torch.rand(n, generator=g) * TMAX
    pred = com_decode(population(te, centers, sig), centers)
    decode_r2 = (1 - ((pred - te) ** 2).sum() / (((te - te.mean()) ** 2).sum() + 1e-9)).item()
    e1 = torch.rand(n, generator=g) * TMAX; e2 = torch.rand(n, generator=g) * TMAX
    sep = (e2 - e1).abs() > 0.15 * TMAX                           # well-separated events
    p1 = com_decode(population(e1, centers, sig), centers)
    p2 = com_decode(population(e2, centers, sig), centers)
    order_acc = (((p2 - p1) > 0) == ((e2 - e1) > 0))[sep].float().mean().item()

    return {"weber_corr_widening": weber_corr_widening, "weber_corr_fixed": weber_corr_fixed,
            "weber_fraction_cv": weber_fraction_cv, "decode_elapsed_r2": decode_r2,
            "order_acc_separated": order_acc}


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=8); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    keys = ["weber_corr_widening", "weber_corr_fixed", "weber_fraction_cv",
            "decode_elapsed_r2", "order_acc_separated"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    lab = {"weber_corr_widening": "Weber: corr(JND, elapsed time), WIDENING fields (->1)",
           "weber_corr_fixed": "  control: same with FIXED-width fields (->0)",
           "weber_fraction_cv": "Weber fraction JND/t, CV in scale-invariant regime (low)",
           "decode_elapsed_r2": "decode elapsed time, R^2 (pop-vector, parameter-free)",
           "order_acc_separated": "event-order accuracy, well-separated events"}
    print(f"TIME CELLS — the temporal axis (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 70, flush=True)
    for k in keys:
        print(f"  {lab[k]:60} {agg[k][0]:.3f} ± {agg[k][1]:.3f}", flush=True)
    print(f"\n  -> widening time fields REPRODUCE scalar/Weber timing (JND grows with elapsed time, corr "
          f"{agg['weber_corr_widening'][0]:.2f}; Weber fraction ~constant, CV {agg['weber_fraction_cv'][0]:.2f}) "
          f"where fixed-width fields do not ({agg['weber_corr_fixed'][0]:.2f});", flush=True)
    print(f"     the population still pinpoints elapsed time (R^2 {agg['decode_elapsed_r2'][0]:.2f}) and "
          f"event order ({agg['order_acc_separated'][0]:.0%}).", flush=True)
    out = {"n_seeds": a.seeds, "TMAX": TMAX, "K": K, "weber": WEBER, "sigma0": SIGMA0,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/time_cells.json", "w"), indent=2)
    svg_time(agg, "results/time_cells.svg")
    print("\nwrote results/time_cells.json and results/time_cells.svg", flush=True)


def svg_time(agg, out):
    centers = torch.linspace(0.0, TMAX, K); sig = fields(centers)          # uniform centers: clean schematic
    ts = torch.linspace(0, TMAX, 200); A = population(ts, centers, sig)
    t_j, jnd_w = jnd_vs_time(centers, fixed=False); _, jnd_f = jnd_vs_time(centers, fixed=True)
    pad = 56; pw = 380; ph = 150
    W = pad + pw + pad; H = 60 + ph + 46 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Time cells: fields tile &amp; widen with latency &#8594; scalar (Weber) timing</text>')
    # top: tuning curves (widening visible)
    oy = 52
    def X(t): return pad + (t / TMAX) * pw
    def Yt(v): return oy + ph - v * (ph - 8)
    for k in range(0, K, 2):
        pts = " ".join(f"{X(ts[i].item()):.1f},{Yt(A[i, k].item()):.1f}" for i in range(0, 200, 2))
        e.append(f'<polyline points="{pts}" fill="none" stroke="hsl({int(20+220*k/K)},70%,45%)" stroke-width="1" opacity="0.65"/>')
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+15:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">elapsed time &#8594; (fields broaden)</text>')
    # bottom: JND vs elapsed time (widening rises = Weber; fixed flat)
    by = oy + ph + 40
    jm = max(jnd_w.max().item(), jnd_f.max().item()) * 1.1
    def Yj(v): return by + ph - (v / jm) * ph
    e.append(f'<text x="{pad}" y="{by-6}" font-size="11" fill="#28324a">timing JND vs elapsed time '
             '(rising = Weber/scalar timing)</text>')
    e.append(f'<line x1="{pad}" y1="{by+ph}" x2="{pad+pw}" y2="{by+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{by}" x2="{pad}" y2="{by+ph}" stroke="#33415c"/>')
    for jnd, col in [(jnd_w, "#2ca25f"), (jnd_f, "#9aa5b8")]:
        pts = " ".join(f"{X(t_j[i].item()):.1f},{Yj(jnd[i].item()):.1f}" for i in range(len(t_j)))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    ly = by + 10
    for col, lab in [("#2ca25f", f"widening fields &#8594; Weber (corr {agg['weber_corr_widening'][0]:.2f})"),
                     ("#9aa5b8", f"fixed-width control (corr {agg['weber_corr_fixed'][0]:.2f})")]:
        e.append(f'<rect x="{pad+pw-176}" y="{ly}" width="13" height="4" fill="{col}"/>')
        e.append(f'<text x="{pad+pw-159}" y="{ly+5}" font-size="9.5" fill="#28324a">{lab}</text>'); ly += 15
    e.append(f'<text x="{pad+pw/2:.0f}" y="{by+ph+15:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">'
             f'elapsed time &#183; decode R&#178;={agg["decode_elapsed_r2"][0]:.2f}, order acc={agg["order_acc_separated"][0]:.0%}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
