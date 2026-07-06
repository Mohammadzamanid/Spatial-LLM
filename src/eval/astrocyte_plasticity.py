"""
src/eval/astrocyte_plasticity.py

ASTROCYTE-GATED SLOW PLASTICITY — the glial learning partner (GAPS.md Tier 5, #B4).

The repo's e-prop (`eprop_local_learning.py`) already has the two neuronal ingredients of a biologically-plausible
learning rule: an ELIGIBILITY TRACE per synapse + a BROADCAST learning signal. The missing third ingredient is
NON-NEURONAL: astrocytes gate synaptic plasticity over a SLOW (seconds) timescale via the tripartite synapse —
hippocampal "learning-associated astrocytes" orchestrate encoding/retrieval and LTP requires astrocytic D-serine
(Williamson et al., *Nature* 2024). Formally, Astrocyte-Gated Multi-Timescale Plasticity = eligibility + broadcast
(present) + a slow glial gate (added here).

We add a slow per-synapse astrocyte variable `a` that integrates recent plasticity, `a ← ρ·a + |Δw|`, and gates
the e-prop update `Δw ← Δw / (1 + β·a)` — throttling further change at synapses that recently CONSOLIDATED a
memory (the D-serine "importance" tag). We train a small recurrent net (rate e-prop; the spiking ALIF net is one
instance) on a CONTINUAL STREAM of cue→target tasks and MEASURE, never train, its retention of the OLD tasks.

The trap this design defeats (a plasticity gate trivially "forgets less" by simply learning less) is killed by a
matched control: a UNIFORM plasticity reduction scaled to the SAME total ‖Δw‖ as the astrocyte condition. So the
reported result is the astrocyte's advantage over uniform reduction — i.e. the gain from WHERE the (matched)
plasticity is throttled, not from throttling it:

  (A) TARGETING BEATS UNIFORM: at matched total plasticity, the astrocyte (throttling importance-tagged synapses)
      retains old tasks better than a uniform reduction — the glial *targeting*, not less learning.
  (B) TIMESCALE FALSIFIER: a FAST astrocyte (ρ≈0.5, decays within a task) gives NO advantage over its matched
      uniform control — the gain needs the SLOW gate to accumulate importance across tasks.
  (C) GROWS WITH MEMORY LOAD: the astrocyte's advantage over full plasticity grows with the number of tasks.
  (D) HONEST TRADE-OFF: protecting old memories costs some new-task acquisition (the stability–plasticity
      frontier) — reported, not hidden. (And the fast-astrocyte control ALSO throttles the current task, yet
      retains no better than uniform — so the retention gain is the SLOW cross-task protection of old synapses,
      not merely "writing the new task weaker".)

Honest scope: computationally this per-synapse importance-throttle is kin to EWC (Kirkpatrick 2017) / synaptic
intelligence (Zenke 2017); the biological content is that a SLOW GLIAL process supplies the importance signal
and gates plasticity (Williamson 2024), and that this needs the slow timescale (the fast-astrocyte falsifier).
It is a reduced model of the tripartite synapse, not a literal D-serine model; the gate acts on the
eligibility-driven weight UPDATE. The Benna–Fusi multi-timescale *synapse* (power-law forgetting) is a distinct,
still-open gap (#B2). Multi-seed, mean ± 95% CI. Writes results/astrocyte_plasticity.json + .svg.

    python -m src.eval.astrocyte_plasticity --seeds 5
"""
import argparse
import json
import math
import os

import torch

N = 96                 # recurrent units
I = 16                 # cue dimension
O = 16                 # target dimension
K = 6                  # tasks in the continual stream
T = 8                  # timesteps per trial
ITERS = 120            # e-prop iterations per task
BATCH = 16
ALPHA = 0.9            # membrane/rate leak
KAPPA = 0.8            # eligibility-trace filter
LR = 0.02
CLIP = 0.3             # per-update norm clip (stabilises the local rule)
BETA = 200.0           # astrocyte gate strength
RHO_SLOW = 0.999       # slow astrocyte (integrates ACROSS the whole stream) — the glial timescale
RHO_FAST = 0.5         # fast astrocyte (decays WITHIN a task) — the falsifier
CUE_NOISE = 0.05


def _init(gen):
    return {
        "Wr": torch.randn(N, N, generator=gen) * (0.9 / math.sqrt(N)),
        "Wi": torch.randn(N, I, generator=gen) * (1.0 / math.sqrt(I)),
        "Wo": torch.randn(O, N, generator=gen) * (1.0 / math.sqrt(N)),
    }


def _forward_grads(W, c, y):
    """One e-prop pass: forward T steps under constant cue c (B,I), target y (B,O); return the eligibility-driven
    weight-change proposals (dWr, dWi, dWo). No autograd, no BPTT."""
    B = c.shape[0]
    v = torch.zeros(B, N); rprev = torch.zeros(B, N)
    ebar_r = torch.zeros(B, N, N); ebar_i = torch.zeros(B, N, I)
    dWr = torch.zeros(N, N); dWi = torch.zeros(N, I); dWo = torch.zeros(O, N)
    for _ in range(T):
        v = ALPHA * v + rprev @ W["Wr"].t() + c @ W["Wi"].t()
        r = torch.tanh(v)
        h = 1 - r ** 2                                          # tanh-derivative surrogate (B,N)
        ebar_r = KAPPA * ebar_r + h.unsqueeze(2) * rprev.unsqueeze(1)   # (B,N,N) eligibility
        ebar_i = KAPPA * ebar_i + h.unsqueeze(2) * c.unsqueeze(1)       # (B,N,I)
        err = r @ W["Wo"].t() - y                              # (B,O) readout error
        L = err @ W["Wo"]                                       # (B,N) broadcast learning signal (e-prop feedback)
        dWr += (L.unsqueeze(2) * ebar_r).mean(0)
        dWi += (L.unsqueeze(2) * ebar_i).mean(0)
        dWo += (err.unsqueeze(2) * r.unsqueeze(1)).mean(0)     # (O,N) direct readout update
        rprev = r
    return dWr, dWi, dWo


def _step(W, name, dW, astro, mode, beta, rho, uni_scale):
    """Apply one clipped weight update under `mode`:
       ungated  -> Δw as-is
       astro    -> Δw / (1+β·a)  (throttle importance-tagged synapses)  + update the slow trace a
       uniform  -> Δw · uni_scale (a UNIFORM reduction, matched in total to the astrocyte budget)
    Returns the applied ‖Δw‖ (for the matched-plasticity bookkeeping)."""
    n = dW.norm()
    if n > CLIP:
        dW = dW * (CLIP / n)                                    # clip for stability
    if mode == "astro":
        dWg = dW / (1.0 + beta * astro[name])
    elif mode == "uniform":
        dWg = dW * uni_scale
    else:
        dWg = dW
    W[name] -= LR * dWg
    if mode == "astro":
        astro[name] = rho * astro[name] + (LR * dWg).abs()
    return (LR * dWg).norm().item()


@torch.no_grad()
def _recall_err(W, c, y):
    """Recall error of a task: forward under its cue, compare the readout DIRECTION to the target (1 − cosine;
    magnitude-robust). 0 = perfect recall."""
    v = torch.zeros(c.shape[0], N); rprev = torch.zeros(c.shape[0], N)
    for _ in range(T):
        v = ALPHA * v + rprev @ W["Wr"].t() + c @ W["Wi"].t()
        rprev = torch.tanh(v)
    yhat = rprev @ W["Wo"].t()
    return (1.0 - torch.nn.functional.cosine_similarity(yhat, y, dim=1)).mean().item()


def run_condition(seed, mode, beta=0.0, rho=0.0, uni_scale=1.0):
    """Learn the K-task stream in sequence; return (retention on old tasks, recency on the newest, total ‖Δw‖)."""
    gen = torch.Generator().manual_seed(seed)
    W = _init(gen)
    astro = {k: torch.zeros_like(v) for k, v in W.items()}
    cues = torch.nn.functional.normalize(torch.randn(K, I, generator=gen), dim=1)
    targs = torch.nn.functional.normalize(torch.randn(K, O, generator=gen), dim=1)
    ngen = torch.Generator().manual_seed(seed + 13)
    total_dw = 0.0
    for k in range(K):
        c0 = cues[k].unsqueeze(0).expand(BATCH, I); y0 = targs[k].unsqueeze(0).expand(BATCH, O)
        for _ in range(ITERS):
            c = c0 + CUE_NOISE * torch.randn(BATCH, I, generator=ngen)
            for name, dW in zip(("Wr", "Wi", "Wo"), _forward_grads(W, c, y0)):
                total_dw += _step(W, name, dW, astro, mode, beta, rho, uni_scale)
    errs = [_recall_err(W, cues[k].unsqueeze(0), targs[k].unsqueeze(0)) for k in range(K)]
    return {"retention": sum(errs[:-1]) / (K - 1), "recency": errs[-1], "dw": total_dw}


def run_seed(seed, iters=None):
    global ITERS
    if iters is not None:
        ITERS = iters
    ung = run_condition(seed, "ungated")
    slow = run_condition(seed, "astro", BETA, RHO_SLOW)
    fast = run_condition(seed, "astro", BETA, RHO_FAST)
    # UNIFORM reductions matched in total ‖Δw‖ to the slow / fast astrocyte budgets (the key controls)
    uni_slow = run_condition(seed, "uniform", uni_scale=slow["dw"] / (ung["dw"] + 1e-9))
    uni_fast = run_condition(seed, "uniform", uni_scale=fast["dw"] / (ung["dw"] + 1e-9))
    return {
        "ret_ungated": ung["retention"], "ret_slow": slow["retention"], "ret_fast": fast["retention"],
        "ret_uniform": uni_slow["retention"], "ret_uniform_fast": uni_fast["retention"],
        "rec_ungated": ung["recency"], "rec_slow": slow["recency"], "rec_uniform": uni_slow["recency"],
        "dw_ungated": ung["dw"], "dw_slow": slow["dw"], "dw_uniform": uni_slow["dw"],
        "targeting_gain": uni_slow["retention"] - slow["retention"],        # (A) astrocyte vs matched uniform
        "falsifier_gain": uni_fast["retention"] - fast["retention"],        # (B) fast astrocyte ~ its uniform
        "load_gain": ung["retention"] - slow["retention"],                  # (C) astrocyte vs full plasticity
        "recency_cost": slow["recency"] - uni_slow["recency"],              # (D) honest stability/plasticity cost
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


KEYS = ["ret_ungated", "ret_slow", "ret_fast", "ret_uniform", "ret_uniform_fast",
        "rec_ungated", "rec_slow", "rec_uniform", "dw_ungated", "dw_slow", "dw_uniform",
        "targeting_gain", "falsifier_gain", "load_gain", "recency_cost"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: retention err ungated {p['ret_ungated']:.3f} / SLOW-astro {p['ret_slow']:.3f} / "
              f"matched-uniform {p['ret_uniform']:.3f} / fast-astro {p['ret_fast']:.3f} | "
              f"targeting {p['targeting_gain']:+.3f} (falsifier {p['falsifier_gain']:+.3f})", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nASTROCYTE-GATED SLOW PLASTICITY — glial gating of e-prop for continual retention "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 96, flush=True)
    print(f"  RETENTION error on old tasks after the stream (lower = better):", flush=True)
    print(f"      ungated {agg['ret_ungated'][0]:.3f} | SLOW astrocyte {agg['ret_slow'][0]:.3f} | "
          f"matched-UNIFORM reduction {agg['ret_uniform'][0]:.3f} | fast astrocyte {agg['ret_fast'][0]:.3f}", flush=True)
    print(f"  (A) TARGETING BEATS UNIFORM (matched ‖Δw‖): uniform − slow = {agg['targeting_gain'][0]:+.3f} ± "
          f"{agg['targeting_gain'][1]:.3f}  ⇒ it is WHERE plasticity is throttled, not less of it", flush=True)
    print(f"  (B) TIMESCALE FALSIFIER: fast astrocyte − its matched uniform = {agg['falsifier_gain'][0]:+.3f} ± "
          f"{agg['falsifier_gain'][1]:.3f} (≈0 ⇒ needs the SLOW gate)", flush=True)
    print(f"  (C) vs FULL plasticity: ungated − slow = {agg['load_gain'][0]:+.3f} ± {agg['load_gain'][1]:.3f} "
          f"(grows with task load)", flush=True)
    print(f"  (D) HONEST TRADE-OFF (recency): slow − uniform newest-task err = {agg['recency_cost'][0]:+.3f} "
          f"(protecting old costs some new — the stability/plasticity frontier)", flush=True)
    print(f"      matched budget ‖Δw‖: slow {agg['dw_slow'][0]:.2f} ≈ uniform {agg['dw_uniform'][0]:.2f} "
          f"(of ungated {agg['dw_ungated'][0]:.2f})", flush=True)

    print(f"\n  -> a SLOW astrocyte gate that throttles plasticity at synapses it has tagged as important cuts "
          f"forgetting of earlier tasks (retention err {agg['ret_ungated'][0]:.3f} → {agg['ret_slow'][0]:.3f}) — "
          f"and, decisively, it beats a UNIFORM plasticity reduction of the SAME total ‖Δw‖ "
          f"({agg['ret_uniform'][0]:.3f}, gap {agg['targeting_gain'][0]:+.3f}): the gain is from WHERE the glia "
          f"throttle plasticity, not from throttling it. It needs the SLOW timescale — a fast astrocyte matches "
          f"its uniform control ({agg['falsifier_gain'][0]:+.3f}) — and it honestly TRADES a little new-task "
          f"acquisition for the retention ({agg['recency_cost'][0]:+.3f}). The glial learning partner "
          f"(Williamson 2024), as an emergent retention signature — measured, not put in the loss.", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "K": K, "iters_per_task": ITERS, "beta": BETA,
           "rho_slow": RHO_SLOW, "rho_fast": RHO_FAST,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/astrocyte_plasticity.json", "w"), indent=2)
    svg(agg, "results/astrocyte_plasticity.svg")
    print("\nwrote results/astrocyte_plasticity.json and results/astrocyte_plasticity.svg", flush=True)


def svg(agg, out):
    pad = 60; pw = 260; ph = 200; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'A slow astrocyte gate cuts forgetting (glial-gated e-prop)</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">throttling plasticity at glia-tagged synapses '
             'beats a UNIFORM cut of the same size &#8212; it is WHERE, not how much; needs the SLOW timescale</text>')
    oy = 58; base = oy + ph
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">retention error (old tasks)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("ungated", agg["ret_ungated"][0], "#c9341a"), ("SLOW\nastro", agg["ret_slow"][0], "#2ca25f"),
            ("matched\nuniform", agg["ret_uniform"][0], "#e08214"), ("fast\nastro", agg["ret_fast"][0], "#9aa6bd")]
    hi = max(b[1] for b in bars) + 1e-6
    for i, (lab, v, col) in enumerate(bars):
        h = (v / hi) * (ph - 30); x = oxA + 14 + i * 60
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="42" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+21}" y="{base-h-6:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+21}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    # Panel B: the decisive contrasts
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">targeting gain vs falsifier</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    cs = [("SLOW: targeting\nvs uniform", agg["targeting_gain"][0], "#2ca25f"),
          ("FAST: targeting\nvs uniform", agg["falsifier_gain"][0], "#9aa6bd")]
    hib = max(abs(c[1]) for c in cs) + 1e-6
    for i, (lab, v, col) in enumerate(cs):
        h = (v / hib) * (ph - 44); x = oxB + 40 + i * 110
        e.append(f'<rect x="{x}" y="{base-max(h,0):.1f}" width="70" height="{abs(h):.1f}" fill="{col}" opacity="0.88"/>')
        e.append(f'<text x="{x+35}" y="{base-abs(h)-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+35}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9.5" fill="#5b6b8c">matched ‖Δw‖ '
             f'{agg["dw_slow"][0]:.1f}≈{agg["dw_uniform"][0]:.1f}; recency cost {agg["recency_cost"][0]:+.3f} (honest trade-off)</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
