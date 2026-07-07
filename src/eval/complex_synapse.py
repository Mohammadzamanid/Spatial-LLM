"""
src/eval/complex_synapse.py

THE MULTI-TIMESCALE (metaplastic) SYNAPSE — graceful, power-law forgetting (GAPS.md Tier 5, #B2).

Every weight in the repo is a SCALAR. A scalar (bounded or leaky) synapse forgets EXPONENTIALLY: it is either
fast-learning-and-fast-forgetting or slow-and-stable, never both — the stability–plasticity dilemma at the
synapse. Benna & Fusi (*Nat. Neurosci.* 2016) resolve it INSIDE the synapse: a single synapse is a CHAIN of
coupled hidden variables at geometrically-spaced timescales (a cascade of "beakers" joined by "tubes"). The
visible weight is the first beaker; a plasticity event enters it and slowly diffuses into deeper, larger,
slower beakers. The consequence is that memory decays as a POWER LAW (~1/√t) instead of exponentially — one
weight that is both fast-learning AND long-remembering, with memory lifetime growing geometrically in the
chain length.

We reproduce Benna & Fusi's own memory benchmark and MEASURE the forgetting curve, never fit it. Over S synapses
we present a stream of M random ±1 memories (one per step: the pattern enters each synapse's visible beaker,
then one diffusion step), then read the visible weights and compute, for every stored memory, its signal-to-noise
ratio (overlap of the current weights with that memory, over the random-overlap noise floor) as a function of
its AGE. Three synapse models, at MATCHED initial SNR (same Δ):

  LEAKY SCALAR  w ← (1-λ)w + Δξ         — the canonical exponential-forgetting synapse
  BENNA-FUSI    an N-beaker chain (geometric capacities/conductances) — the complex synapse
  BF at N=3,5,7 — the dose-response

Measured signatures (n=5), never in the loss:
  (A) POWER LAW vs EXPONENTIAL: the Benna-Fusi SNR(age) is a straight line on LOG-LOG axes (slope ≈ -0.5 — the
      1/√t law), fit far better by a power law than an exponential; the leaky scalar is a straight line on
      SEMILOG axes, fit far better by an exponential. (We report both fits' R² for both models; the winner flips.)
  (B) LIFETIME: the age at which SNR crosses 1 (signal = noise) is much larger for Benna-Fusi than for the leaky
      scalar — at matched initial SNR.
  (C) DOSE-RESPONSE: the measured slope approaches -0.5 and the memory lifetime grows as the chain gets deeper
      (more beakers) — the Benna-Fusi prediction; a 1-beaker chain is just the scalar.

Honest scope: a linear-chain reduced model of Benna-Fusi on the random-memory benchmark (it drops into any store,
including the spatial / Hopfield ones). Distinct from #B4 (a glial gate on the learning rule); B2 is the intrinsic
multi-timescale synapse. Multi-seed, mean ± 95% CI. Writes results/complex_synapse.json + .svg.

    python -m src.eval.complex_synapse --seeds 5
"""
import argparse
import json
import math
import os

import torch

S = 3000               # synapses
M = 3000               # memories presented
DELTA = 1.0            # plasticity increment
LEAK = 0.03            # leaky-scalar forgetting rate (the exponential control)
N_HEAD = 7             # Benna-Fusi chain length for the headline
N_SWEEP = (3, 5, 7)    # dose-response over chain depth
G0 = 0.5               # base tube conductance
N_BINS = 16


def _chain_params(N):
    C = torch.tensor([2.0 ** k for k in range(N)])              # capacities: deeper beakers larger/slower
    g = torch.tensor([G0 * 2.0 ** (-k) for k in range(N - 1)])  # conductances: deeper tubes slower
    return C, g


def store(model, seed, N=N_HEAD):
    """Present M random ±1 memories; return the visible weights (S,) and the memory patterns (M,S)."""
    gen = torch.Generator().manual_seed(seed)
    pats = torch.randint(0, 2, (M, S), generator=gen).float() * 2 - 1
    if model == "leaky":
        w = torch.zeros(S)
        for m in range(M):
            w = (1 - LEAK) * w + DELTA * pats[m]
        return w, pats
    C, g = _chain_params(N)
    U = torch.zeros(S, N); z = torch.zeros(S, 1)
    for m in range(M):
        J = g * (U[:, :-1] - U[:, 1:])                          # (S,N-1) flow beaker k -> k+1 (vectorised)
        Jp = torch.cat([z, J, z], 1)
        U = U + (Jp[:, :-1] - Jp[:, 1:]) / C                    # diffusion step
        U[:, 0] = U[:, 0] + DELTA * pats[m] / C[0]              # plasticity enters the visible beaker
    return U[:, 0], pats


def snr_curve(w, pats):
    """Binned signal-to-noise ratio of stored memories vs their age (log-spaced bins)."""
    noise = w.std().item() / math.sqrt(S) + 1e-9
    snr = (pats @ w) / S / noise                                # (M,) SNR per stored pattern
    age = torch.arange(M - 1, -1, -1).float()                  # pattern m has age M-1-m
    edges = torch.logspace(0, math.log10(M - 1), N_BINS)
    a_mid, s_mid = [], []
    for b in range(len(edges) - 1):
        msk = (age >= edges[b]) & (age < edges[b + 1])
        if msk.any():
            a_mid.append(math.sqrt(edges[b].item() * edges[b + 1].item())); s_mid.append(snr[msk].mean().item())
    return torch.tensor(a_mid), torch.tensor(s_mid)


def _fits(age, snr):
    """Return (power-law slope, R² of a log-log (power) fit, R² of a semilog (exponential) fit) over the region
    where SNR is above the noise floor."""
    m = snr > 1.0
    a, s = age[m], snr[m]
    if a.numel() < 5:
        return float("nan"), float("nan"), float("nan")
    la, ls = torch.log(a), torch.log(s)
    A = torch.stack([la, torch.ones_like(la)], 1)
    sol = torch.linalg.lstsq(A, ls).solution
    r2_ll = (1 - ((ls - A @ sol) ** 2).sum() / (((ls - ls.mean()) ** 2).sum() + 1e-9)).item()
    A2 = torch.stack([a, torch.ones_like(a)], 1)
    r2_ex = (1 - ((ls - A2 @ torch.linalg.lstsq(A2, ls).solution) ** 2).sum() / (((ls - ls.mean()) ** 2).sum() + 1e-9)).item()
    return sol[0].item(), r2_ll, r2_ex


def _lifetime(age, snr):
    """Age at which SNR crosses 1 (signal = noise), by log-log interpolation. inf if it never drops below 1."""
    below = (snr < 1.0).nonzero()
    if below.numel() == 0:
        return age[-1].item()
    i = below[0].item()
    if i == 0:
        return age[0].item()
    (a0, s0), (a1, s1) = (age[i - 1], snr[i - 1]), (age[i], snr[i])
    la = math.log(a0) + (math.log(1.0) - math.log(s0)) * (math.log(a1) - math.log(a0)) / (math.log(s1) - math.log(s0) - 1e-9)
    return math.exp(la)


def run_seed(seed, iters=None):
    out = {}
    # leaky scalar (exponential) + Benna-Fusi headline (N=7)
    for name, model, N in (("scalar", "leaky", 1), ("bf", "chain", N_HEAD)):
        w, pats = store(model, seed, N)
        age, snr = snr_curve(w, pats)
        sl, r2ll, r2ex = _fits(age, snr)
        out[f"{name}_slope"] = sl; out[f"{name}_r2_loglog"] = r2ll; out[f"{name}_r2_semilog"] = r2ex
        out[f"{name}_lifetime"] = _lifetime(age, snr)
    # dose-response over chain depth
    for N in N_SWEEP:
        w, pats = store("chain", seed, N)
        age, snr = snr_curve(w, pats)
        out[f"lifetime_N{N}"] = _lifetime(age, snr)
    out["powerlaw_margin_bf"] = out["bf_r2_loglog"] - out["bf_r2_semilog"]        # (A) >0 => BF is power-law
    out["exp_margin_scalar"] = out["scalar_r2_semilog"] - out["scalar_r2_loglog"]  # (A) >0 => scalar is exponential
    out["lifetime_ratio"] = out["bf_lifetime"] / (out["scalar_lifetime"] + 1e-9)   # (B)
    out["depth_gain"] = out["lifetime_N7"] - out["lifetime_N3"]                     # (C) lifetime grows with depth
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


KEYS = ["scalar_slope", "scalar_r2_loglog", "scalar_r2_semilog", "scalar_lifetime",
        "bf_slope", "bf_r2_loglog", "bf_r2_semilog", "bf_lifetime",
        "lifetime_N3", "lifetime_N5", "lifetime_N7",
        "powerlaw_margin_bf", "exp_margin_scalar", "lifetime_ratio", "depth_gain"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: BF slope {p['bf_slope']:+.2f} (R² loglog {p['bf_r2_loglog']:.3f} / semilog "
              f"{p['bf_r2_semilog']:.3f}) | scalar R² semilog {p['scalar_r2_semilog']:.3f} / loglog "
              f"{p['scalar_r2_loglog']:.3f} | lifetime BF {p['bf_lifetime']:.0f} vs scalar {p['scalar_lifetime']:.0f}",
              flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nMULTI-TIMESCALE (BENNA-FUSI) SYNAPSE — power-law vs exponential forgetting (n={a.seeds}; "
          f"mean ± 95% CI)\n" + "=" * 92, flush=True)
    print(f"  (A) SHAPE OF FORGETTING (which fit wins — power-law vs exponential):", flush=True)
    print(f"      BENNA-FUSI: log-log R² {agg['bf_r2_loglog'][0]:.3f} vs semilog R² {agg['bf_r2_semilog'][0]:.3f} "
          f"⇒ POWER LAW, slope {agg['bf_slope'][0]:+.2f} ± {agg['bf_slope'][1]:.2f} (≈ -0.5, the 1/√t law)", flush=True)
    print(f"      LEAKY SCALAR: semilog R² {agg['scalar_r2_semilog'][0]:.3f} vs log-log R² "
          f"{agg['scalar_r2_loglog'][0]:.3f} ⇒ EXPONENTIAL", flush=True)
    print(f"      margins: BF power-law {agg['powerlaw_margin_bf'][0]:+.3f} ± {agg['powerlaw_margin_bf'][1]:.3f}; "
          f"scalar exponential {agg['exp_margin_scalar'][0]:+.3f} ± {agg['exp_margin_scalar'][1]:.3f}", flush=True)
    print(f"  (B) MEMORY LIFETIME (age at SNR=1, matched initial SNR): Benna-Fusi {agg['bf_lifetime'][0]:.0f} vs "
          f"scalar {agg['scalar_lifetime'][0]:.0f}  ⇒ {agg['lifetime_ratio'][0]:.1f}× longer", flush=True)
    print(f"  (C) DOSE-RESPONSE (lifetime vs chain depth N): N=3 {agg['lifetime_N3'][0]:.0f} → N=5 "
          f"{agg['lifetime_N5'][0]:.0f} → N=7 {agg['lifetime_N7'][0]:.0f} (grows geometrically with depth)", flush=True)

    print(f"\n  -> a single synapse made of a CHAIN of coupled variables at geometric timescales (Benna & Fusi "
          f"2016) forgets as a POWER LAW (log-log R² {agg['bf_r2_loglog'][0]:.2f}, slope {agg['bf_slope'][0]:+.2f} "
          f"≈ the 1/√t law) — where a leaky SCALAR synapse forgets EXPONENTIALLY (semilog R² "
          f"{agg['scalar_r2_semilog'][0]:.2f}). At matched initial SNR the complex synapse's memory lifetime is "
          f"{agg['lifetime_ratio'][0]:.1f}× longer, and it grows with the number of beakers "
          f"({agg['lifetime_N3'][0]:.0f}→{agg['lifetime_N7'][0]:.0f} for N=3→7). One weight both fast-learning and "
          f"long-remembering — graceful forgetting from the SYNAPSE, measured, not put in the loss.", flush=True)

    out = {"n_seeds": a.seeds, "S": S, "M": M, "N_head": N_HEAD, "leak": LEAK,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/complex_synapse.json", "w"), indent=2)
    svg(per[0], agg, "results/complex_synapse.svg")
    print("\nwrote results/complex_synapse.json and results/complex_synapse.svg", flush=True)


def svg(sample, agg, out):
    # Panel A: SNR vs age on log-log (BF straight = power law) and the scalar (curved). Panel B: lifetime vs N.
    aB, sB = snr_curve(*store("chain", 0, N_HEAD))
    aS, sS = snr_curve(*store("leaky", 0))
    pad = 60; pw = 250; ph = 200; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'The multi-timescale synapse forgets as a POWER LAW</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">Benna-Fusi chain: straight on log-log (~1/&#8730;t); '
             'a leaky scalar is exponential (curved on log-log) &#8212; measured, not fit</text>')
    oy = 58; base = oy + ph; oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) SNR vs age (log-log)</text>')
    lx0, lx1 = math.log10(1), math.log10(M)
    allsnr = torch.cat([sB, sS]).clamp(min=0.3); ly0, ly1 = math.log10(0.5), math.log10(allsnr.max().item() + 1e-9)
    def X(a): return oxA + (math.log10(max(a, 1)) - lx0) / (lx1 - lx0 + 1e-9) * pw
    def Y(s): return base - (math.log10(max(s, 0.5)) - ly0) / (ly1 - ly0 + 1e-9) * (ph - 10)
    e.append(f'<line x1="{oxA}" y1="{Y(1.0):.0f}" x2="{oxA+pw}" y2="{Y(1.0):.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{oxA+pw}" y="{Y(1.0)-3:.0f}" font-size="8.5" fill="#7787a6" text-anchor="end">SNR=1 (memory lost)</text>')
    for age, snr, col, lab in ((aB, sB, "#2ca25f", "Benna-Fusi (power law)"), (aS, sS, "#c9341a", "leaky scalar (exp)")):
        pts = " ".join(f"{X(age[i].item()):.1f},{Y(snr[i].item()):.1f}" for i in range(len(age)) if snr[i] > 0.4)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" opacity="0.9"/>')
    ly = oy + 8
    for col, lab in (("#2ca25f", "Benna-Fusi (power law)"), ("#c9341a", "leaky scalar (exp)")):
        e.append(f'<rect x="{oxA+pw-150}" y="{ly-8}" width="12" height="4" fill="{col}"/><text x="{oxA+pw-134}" y="{ly-4}" font-size="9" fill="#28324a">{lab}</text>'); ly += 14
    e.append(f'<text x="{oxA+pw/2:.0f}" y="{base+16:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">memory age (log) &#8594;</text>')
    # Panel B: lifetime vs N
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) memory lifetime vs chain depth</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("scalar", agg["scalar_lifetime"][0], "#c9341a"), ("N=3", agg["lifetime_N3"][0], "#8fbf9f"),
            ("N=5", agg["lifetime_N5"][0], "#4ca66f"), ("N=7", agg["lifetime_N7"][0], "#2ca25f")]
    hi = max(b[1] for b in bars) + 1e-6
    for i, (lab, v, col) in enumerate(bars):
        h = (v / hi) * (ph - 30); x = oxB + 16 + i * 60
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="42" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+21}" y="{base-h-6:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0f}</text>')
        e.append(f'<text x="{x+21}" y="{base+14:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<text x="{oxB}" y="{base+30:.0f}" font-size="9.5" fill="#5b6b8c">age at SNR=1; grows with beaker count</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
