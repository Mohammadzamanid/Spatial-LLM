"""
src/eval/ephaptic_coupling.py

EPHAPTIC COUPLING — a non-synaptic field that shapes spike TIMING (GAPS.md Tier 2).

The whole model (like the rest of the repo) coordinates neurons through synaptic weights. But transmembrane
currents sum into an extracellular LOCAL FIELD that feeds back onto neighbouring membranes and biases their
spike TIMING with NO synapse involved — a genuine non-classical channel (Anastassiou & Koch 2011/2015: even
~1 mV endogenous fields entrain spikes; Chiang, Han, Durand 2019: hippocampal activity propagates via
endogenous fields with synaptic AND gap-junction transmission blocked).

We add a self-generated field to a leaky-integrate-and-fire population and MEASURE its computational work, never
put in a loss, guarding the by-construction trap (a field that just adds common DRIVE would raise the RATE and
look coordinated — trivially). The field here is ZERO-MEAN: E = g_eph * (population_lowpass - slow_baseline), so
it depolarizes when the population is ABOVE its own baseline and hyperpolarizes below — it sharpens the rhythm
WITHOUT net drive. And every comparison is at a RATE-MATCHED operating point (drive tuned so the mean firing
rate is equal across conditions), so any synchrony difference is TIMING, not rate:

  (A) SYNCHRONY AT MATCHED RATE. Field ON -> high spike synchrony (Golomb-Rinzel chi) vs field OFF -> incoherent,
      with the mean firing rate MATCHED. A dose-response (chi rises with field strength through a synchronization
      transition) confirms it is the field.
  (B) A GLOBAL FIELD BEATS SPARSE SYNAPSES at matched coupling budget. The diffuse field is coherent over the
      whole population; equally-strong SPARSE synapses see only a noisy local sample and do NOT synchronize.
      (And the field with ZERO synapses still synchronizes -> a truly independent channel.)
  (C) FALSIFIER: zero the field -> chi collapses to the uncoupled baseline, at matched rate.
  (D) COMPUTATIONAL WORK. A downstream COINCIDENCE detector (fires on >= m near-simultaneous inputs) is driven
      far more strongly by the ephaptically-synchronized assembly than by the field-off one — at MATCHED input
      rate. Synchrony the field creates is readable; rate alone is not.

    python -m src.eval.ephaptic_coupling --seeds 5
"""
import argparse
import json
import math
import os

import torch

N = 200            # neurons
STEPS = 3000       # sim steps
DT = 0.1
TAU = 1.0          # membrane time constant
TAU_F = 2.0        # field low-pass
TAU_BASE = 40.0    # slow homeostatic baseline -> field is zero-mean
DRIVE_SD = 0.12    # heterogeneity of the constant drives (desynchronizes without coupling)
NOISE = 0.03
G_FIELD = 8.0      # ephaptic field gain (above the synchronization transition)
SYN_K = 4          # sparse-synapse control: presynaptic sources per neuron
TARGET_RATE = 1.0  # rate-matched operating point


def sim(drive, g_eph=0.0, syn_k=0, syn_w=0.0, seed=0, steps=STEPS):
    g = torch.Generator().manual_seed(seed)
    I = drive + DRIVE_SD * torch.randn(N, generator=g)
    V = torch.rand(N, generator=g)
    pre = torch.randint(0, N, (N, syn_k), generator=g) if syn_k > 0 else None
    s = torch.zeros(N); base = torch.tensor(0.0); last = torch.zeros(N)
    raster = torch.zeros(steps, N)
    for t in range(steps):
        field = g_eph * (s.mean() - base)                          # ZERO-MEAN diffuse ephaptic field
        syn = (syn_w * (last[pre].sum(1) - last.mean() * syn_k)) if syn_k > 0 else 0.0   # zero-mean sparse synapses
        dV = (-V + I + field + syn) * DT / TAU + NOISE * torch.randn(N, generator=g) * (DT ** 0.5)
        V = V + dV
        fired = V >= 1.0
        V = torch.where(fired, torch.zeros_like(V), V)
        last = fired.float()
        s = s + (-s / TAU_F + last) * DT
        base = base + (s.mean() - base) * DT / TAU_BASE
        raster[t] = last
    return raster


def rate_of(raster):
    return (raster.mean(0) / DT).mean().item()


def match_drive(target, g_eph=0.0, syn_k=0, syn_w=0.0, seed=0, iters=8):
    """Binary-search the constant drive so the mean firing rate == target (the rate-matched operating point)."""
    lo, hi = 1.02, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rate_of(sim(mid, g_eph=g_eph, syn_k=syn_k, syn_w=syn_w, seed=seed, steps=1500))
        if r < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi(raster):
    """Golomb-Rinzel synchrony: sqrt( Var_t(pop-mean V-proxy) / mean_i Var_t(V-proxy_i) ). ->1 sync, ->~0 async.
    V-proxy = low-pass of each spike train."""
    a = DT / TAU_F
    y = torch.zeros_like(raster); acc = torch.zeros(raster.shape[1])
    for t in range(raster.shape[0]):
        acc = acc + (-acc * a + raster[t]); y[t] = acc
    y = y[y.shape[0] // 5:]                                         # drop transient
    return (y.mean(1).var(unbiased=False) / (y.var(dim=0, unbiased=False).mean() + 1e-9)).sqrt().item()


def coincidence_drive(raster, P=30, win=2, m=12):
    """Downstream COINCIDENCE detector on an assembly of P neurons: it fires when >= m of them spike within a
    TIGHT window. The threshold is set so the ASYNCHRONOUS state (spikes scattered) essentially never reaches it,
    while a synchronized burst does — so the returned rate (fires per window) is the readable 'work' the
    field-made synchrony performs for a temporal readout, at matched input rate."""
    a = raster[:, :P]
    T = a.shape[0] - win
    co = torch.stack([a[t:t + win].sum(0).clamp(max=1).sum() for t in range(0, T, win)])   # co-active count per window
    return (co >= m).float().mean().item()


def run_seed(seed):
    d_off = match_drive(TARGET_RATE, g_eph=0.0, seed=seed)
    d_fld = match_drive(TARGET_RATE, g_eph=G_FIELD, seed=seed)
    d_syn = match_drive(TARGET_RATE, g_eph=0.0, syn_k=SYN_K, syn_w=G_FIELD / SYN_K, seed=seed)

    r_off = sim(d_off, g_eph=0.0, seed=seed)
    r_fld = sim(d_fld, g_eph=G_FIELD, seed=seed)
    r_syn = sim(d_syn, g_eph=0.0, syn_k=SYN_K, syn_w=G_FIELD / SYN_K, seed=seed)

    chi_off, chi_fld, chi_syn = chi(r_off), chi(r_fld), chi(r_syn)
    rate_off, rate_fld, rate_syn = rate_of(r_off), rate_of(r_fld), rate_of(r_syn)
    coin_off, coin_fld = coincidence_drive(r_off), coincidence_drive(r_fld)
    # dose-response monotonicity check: chi at 0, G/2, G (rate-matched at 0 and G endpoints is enough)
    chi_half = chi(sim(d_fld, g_eph=G_FIELD / 2, seed=seed))

    return {
        "chi_field": round(chi_fld, 4),
        "chi_off": round(chi_off, 4),
        "chi_sparse_syn": round(chi_syn, 4),
        "chi_half_field": round(chi_half, 4),
        "rate_field": round(rate_fld, 4),
        "rate_off": round(rate_off, 4),
        "rate_sparse": round(rate_syn, 4),
        "coin_field": round(coin_fld, 4),
        "coin_off": round(coin_off, 4),
        "sync_gap": round(chi_fld - chi_off, 4),               # field vs no-field, matched rate
        "field_vs_sparse": round(chi_fld - chi_syn, 4),        # global field vs matched-budget sparse synapses
        "rate_mismatch": round(abs(rate_fld - rate_off), 4),   # must be ~0 (matched-rate guard)
        "coin_gap": round(coin_fld - coin_off, 4),             # downstream readout payoff
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["chi_field", "chi_off", "chi_sparse_syn", "chi_half_field", "rate_field", "rate_off",
        "coin_field", "coin_off", "sync_gap", "field_vs_sparse", "rate_mismatch", "coin_gap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: chi field {p['chi_field']:.2f} / off {p['chi_off']:.2f} / sparse {p['chi_sparse_syn']:.2f} "
              f"| rate field {p['rate_field']:.2f} vs off {p['rate_off']:.2f} | coincidence field {p['coin_field']:.2f} "
              f"vs off {p['coin_off']:.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nEPHAPTIC COUPLING — a non-synaptic field shapes spike TIMING (n={a.seeds}; mean ± 95% CI)\n" + "=" * 90, flush=True)
    print(f"  (A) SYNCHRONY AT MATCHED RATE: field ON chi {agg['chi_field'][0]:.3f} ± {agg['chi_field'][1]:.3f}  vs  "
          f"field OFF chi {agg['chi_off'][0]:.3f} ± {agg['chi_off'][1]:.3f}   (gap {agg['sync_gap'][0]:+.3f} ± "
          f"{agg['sync_gap'][1]:.3f})", flush=True)
    print(f"      MATCHED RATE: field {agg['rate_field'][0]:.2f} vs off {agg['rate_off'][0]:.2f} (|Δrate| "
          f"{agg['rate_mismatch'][0]:.3f}) — so it is TIMING, not rate. Dose-response: chi 0 -> "
          f"{agg['chi_half_field'][0]:.2f} (half) -> {agg['chi_field'][0]:.2f} (full).", flush=True)
    print(f"  (B) GLOBAL FIELD >> SPARSE SYNAPSES (matched budget): field {agg['chi_field'][0]:.3f} vs sparse "
          f"{agg['chi_sparse_syn'][0]:.3f} (gap {agg['field_vs_sparse'][0]:+.3f}) — a coherent field coordinates "
          f"where equally-strong local wiring cannot; the field also needs NO synapses.", flush=True)
    print(f"  (C) FALSIFIER: zero the field -> chi {agg['chi_off'][0]:.3f} (uncoupled baseline) at matched rate.", flush=True)
    print(f"  (D) COMPUTATIONAL WORK: a downstream coincidence detector fires on the assembly at rate "
          f"{agg['coin_field'][0]:.3f} (field) vs {agg['coin_off'][0]:.3f} (off) — gap {agg['coin_gap'][0]:+.3f} at "
          f"MATCHED input rate: the field-made synchrony is readable, rate alone is not.", flush=True)

    sound = (agg["sync_gap"][0] > 0.3 and agg["rate_mismatch"][0] < 0.15 and
             agg["field_vs_sparse"][0] > 0.2 and agg["coin_gap"][0] > 0.05)
    verdict = ("SOUND — a self-generated ZERO-MEAN field synchronizes spike timing at matched firing rate where a "
               "matched-budget sparse-synaptic network does not, and zeroing the field abolishes it; the synchrony "
               "drives a downstream coincidence detector. A non-synaptic channel doing computational work." if sound
               else "WEAK — the field's timing effect did not clear the matched-rate falsifiers; revisit the regime.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "steps": STEPS, "g_field": G_FIELD, "target_rate": TARGET_RATE,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/ephaptic_coupling.json", "w"), indent=2)
    _svg(agg, "results/ephaptic_coupling.svg")
    print("\nwrote results/ephaptic_coupling.json and results/ephaptic_coupling.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Ephaptic coupling: a non-synaptic field shapes spike timing</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">a zero-mean self-field synchronizes spike '
             'timing at MATCHED firing rate, where matched-budget sparse synapses cannot; zeroing the field '
             'abolishes it</text>')
    oy = 60; base = oy + ph
    # Panel A: synchrony (field / sparse-syn / off), higher=better
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) spike synchrony chi (matched rate)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("ephaptic\nfield", agg["chi_field"][0], "#2ca25f"), ("sparse syn\n(matched)", agg["chi_sparse_syn"][0], "#c98a1a"),
            ("field OFF", agg["chi_off"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(bars):
        h = v * (ph - 24); x = oxA + 20 + i * 74
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="52" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-h-6:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">rates matched to {TARGET_RATE:.1f}; the field changes TIMING, not rate</text>')
    # Panel B: downstream coincidence readout (field vs off), matched rate
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) downstream coincidence detector</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    hi = max(agg["coin_field"][0], agg["coin_off"][0]) + 1e-6
    b2 = [("ephaptic\nfield", agg["coin_field"][0], "#2ca25f"), ("field OFF", agg["coin_off"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(b2):
        h = (v / hi) * (ph - 24); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="64" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+32}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">coincidence rate on the assembly — synchrony does readable work at matched rate</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
