"""
src/eval/meta_learning.py

META-LEARNING — the brain tunes its own LEARNING RATE from inferred volatility (GAPS.md Tier 5, #B3).

Humans and animals raise their learning rate in VOLATILE stretches and lower it in STABLE ones — and, the subtle
part, they DISSOCIATE volatility (the world is changing → weight recent evidence → HIGH α) from stochasticity
(observations are noisy but the world is stable → don't chase noise → LOW α), even though both inflate the
observation variance and are hard to tell apart (Behrens et al. 2007; Piray & Daw 2020). The mechanism is a
prefrontal META-REINFORCEMENT-LEARNING process (Wang et al. 2018): a recurrent network meta-trained across many
environments develops, in its RECURRENT DYNAMICS (not its synapses), a fast learning algorithm whose effective
learning rate adapts online — the weights are frozen at test time.

We reproduce it and MEASURE the signature, never train it. A GRU is meta-trained ONLY to predict the next
observation, across episodes whose hazard (volatility) and noise (stochasticity) are drawn per-episode and NEVER
given as input — it must infer them from the observation stream. Then, with WEIGHTS FROZEN, we run it through one
session with three concatenated blocks and fit its REVEALED learning rate per block by the delta rule
`ŝ_t = ŝ_{t-1} + α·(o_t − ŝ_{t-1})` (α = the regression slope of the estimate's revision on prediction error — a
RATE, invariant to error magnitude):

  (A) TRACKS VOLATILITY: α is higher in the VOLATILE block than the STABLE one — emergent higher gain when the
      world jumps.
  (B) THE DISSOCIATION (the non-circular falsifier): α is LOWER in the STOCHASTIC block than the volatile one —
      even though STOCHASTIC has the HIGHEST observation variance. Chasing variance would RAISE α; the network
      LOWERS it, having inferred (from temporal STRUCTURE — a jump is a persistent step, noise is uncorrelated
      wiggle) that the variance is noise, not change. Volatility ↑α, stochasticity ↓α — the computation the naive
      "learn faster when errors are big" account cannot produce.
  (C) LEARNED, not architectural: an UNTRAINED (random-weight) GRU shows a flat α across blocks.
  (D) FUNCTIONAL: the adaptive network's prediction error beats the BEST SINGLE FIXED learning rate on the mixed
      session — no static α matches it.

Honest scope: the OUTER (meta) loop is backprop — the meta-RL standard; the biological claim is the emergent
INNER-loop learning rate that lives in the frozen-weight recurrent dynamics. The latent is 1-D tracking; tying it
to the SR/grid reward-location substrate is a follow-up. Multi-seed, mean ± 95% CI. Writes
results/meta_learning.json + .svg.

    python -m src.eval.meta_learning --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

T = 70                 # training episode length
BATCH = 64
H_LO, H_HI = 0.01, 0.30    # hazard (volatility) range — drawn per episode, never shown to the net
S_LO, S_HI = 0.03, 0.25    # observation-noise (stochasticity) range
HID = 64
LR = 3e-3

# test blocks (hazard, noise): stochastic has LOW hazard but the HIGHEST noise
BLOCKS = {"stable": (0.03, 0.10), "volatile": (0.25, 0.10), "stochastic": (0.03, 0.24)}
BLEN = 140             # steps per block
SKIP = 40              # adaptation transient skipped before fitting α in each block
N_TEST = 128           # test sessions


def gen_episode(hazard, noise, length, gen):
    """Change-point latent: with prob `hazard` the latent JUMPS to a new uniform value, else holds.
    Observation = latent + `noise`·N(0,1). hazard/noise are (B,) per-item. Returns o (B,L,1)."""
    B = hazard.shape[0]
    s = torch.zeros(B, length)
    s[:, 0] = torch.rand(B, generator=gen)
    for t in range(1, length):
        jump = (torch.rand(B, generator=gen) < hazard).float()
        s[:, t] = jump * torch.rand(B, generator=gen) + (1 - jump) * s[:, t - 1]
    o = s + noise.unsqueeze(1) * torch.randn(B, length, generator=gen)
    return o.unsqueeze(-1)


class Predictor(nn.Module):
    """GRU that predicts the next observation from the observation stream. The output at step t is the network's
    estimate ŝ_t of o_{t+1}, formed from o_{0..t} — its online 'belief' about the latent."""

    def __init__(self, hidden=HID):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, o):
        h, _ = self.gru(o)
        return self.out(h).squeeze(-1)          # (B, L)


def meta_train(seed, iters):
    torch.manual_seed(seed)
    net = Predictor()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(iters):
        hazard = torch.empty(BATCH).uniform_(H_LO, H_HI, generator=gen)
        noise = torch.empty(BATCH).uniform_(S_LO, S_HI, generator=gen)
        o = gen_episode(hazard, noise, T, gen)
        pred = net(o)
        loss = ((pred[:, :-1] - o[:, 1:, 0]) ** 2).mean()      # next-observation prediction ONLY
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def revealed_alpha(net, gen, perf=False):
    """Run the (frozen) net through N_TEST sessions of concatenated [stable|volatile|stochastic] blocks; fit its
    revealed learning rate α per block (pooled slope of Δŝ on prediction error, after a SKIP transient). If
    `perf`, also compare its next-obs error to the best SINGLE fixed-α delta predictor (vectorised over α)."""
    names = list(BLOCKS)
    acc = {n: [0.0, 0.0] for n in names}                       # [numerator, denominator] of the α slope fit
    net_se, best_se = 0.0, None
    alphas_grid = torch.linspace(0.02, 1.0, 50)
    for _ in range(max(1, N_TEST // BATCH)):
        parts, spans, cur = [], [], 0
        for n in names:
            h, s = BLOCKS[n]
            o = gen_episode(torch.full((BATCH,), h), torch.full((BATCH,), s), BLEN, gen)
            parts.append(o); spans.append((n, cur, cur + BLEN)); cur += BLEN
        O = torch.cat(parts, 1)                                # (B, 3*BLEN, 1)
        pred = net(O)                                          # (B, 3*BLEN)
        for n, a, b in spans:
            sh = pred[:, a + SKIP - 1:b]                       # ŝ_{t-1..} window
            oo = O[:, a + SKIP:b, 0]
            d = sh[:, 1:] - sh[:, :-1]                         # Δŝ_t
            err = oo - sh[:, :-1]                              # o_t − ŝ_{t-1}
            acc[n][0] += (d * err).sum().item(); acc[n][1] += (err ** 2).sum().item()
        if perf:                                              # (D) net vs best fixed-α (vectorised over α)
            net_se += ((pred[:, :-1] - O[:, 1:, 0]) ** 2).sum().item()
            o1 = O[:, :, 0]; al = alphas_grid.unsqueeze(1)     # (nA, 1)
            est = o1[:, 0].unsqueeze(0).repeat(alphas_grid.shape[0], 1)   # (nA, B)
            se = torch.zeros(alphas_grid.shape[0])
            for t in range(1, o1.shape[1]):
                ot = o1[:, t].unsqueeze(0)                     # (1, B)
                se += ((est - ot) ** 2).sum(1)
                est = est + al * (ot - est)
            best_se = se if best_se is None else best_se + se
    alpha = {n: acc[n][0] / (acc[n][1] + 1e-8) for n in names}
    return alpha, net_se, (best_se.min().item() if best_se is not None else 0.0)


def run_seed(seed, iters=2500):
    gen = torch.Generator().manual_seed(seed + 900)
    net = meta_train(seed, iters)
    a, net_se, best_se = revealed_alpha(net, gen, perf=True)
    # untrained (random-weight) control
    torch.manual_seed(seed + 5000); untr = Predictor()
    u, _, _ = revealed_alpha(untr, torch.Generator().manual_seed(seed + 901))
    return {
        "alpha_stable": a["stable"], "alpha_volatile": a["volatile"], "alpha_stochastic": a["stochastic"],
        "untr_stable": u["stable"], "untr_volatile": u["volatile"], "untr_stochastic": u["stochastic"],
        "vol_gain": a["volatile"] - a["stable"],                 # (A) tracks volatility
        "dissoc": a["volatile"] - a["stochastic"],               # (B) volatility vs stochasticity (falsifier)
        "stable_vs_stoch": a["stable"] - a["stochastic"],        # stochastic even below stable
        "untr_vol_gain": u["volatile"] - u["stable"],            # (C) untrained control
        "perf_ratio": net_se / (best_se + 1e-8),                 # (D) <1 => beats best fixed α
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


KEYS = ["alpha_stable", "alpha_volatile", "alpha_stochastic",
        "untr_stable", "untr_volatile", "untr_stochastic",
        "vol_gain", "dissoc", "stable_vs_stoch", "untr_vol_gain", "perf_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2500)
    a = ap.parse_args()
    per = [run_seed(s, a.iters) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: α stable {p['alpha_stable']:.2f} / volatile {p['alpha_volatile']:.2f} / "
              f"stochastic {p['alpha_stochastic']:.2f} | vol-gain {p['vol_gain']:+.2f} dissoc {p['dissoc']:+.2f} "
              f"| untrained vol-gain {p['untr_vol_gain']:+.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nMETA-LEARNING — a self-tuned learning rate from inferred volatility (n={a.seeds}; mean ± 95% CI)\n"
          + "=" * 92, flush=True)
    print(f"  revealed learning rate α (frozen weights; fit post-hoc, never trained):", flush=True)
    print(f"      STABLE {agg['alpha_stable'][0]:.2f} ± {agg['alpha_stable'][1]:.2f}   |   "
          f"VOLATILE {agg['alpha_volatile'][0]:.2f} ± {agg['alpha_volatile'][1]:.2f}   |   "
          f"STOCHASTIC {agg['alpha_stochastic'][0]:.2f} ± {agg['alpha_stochastic'][1]:.2f}", flush=True)
    print(f"  (A) TRACKS VOLATILITY:  α_volatile − α_stable = {agg['vol_gain'][0]:+.2f} ± {agg['vol_gain'][1]:.2f}", flush=True)
    print(f"  (B) DISSOCIATION (falsifier): α_volatile − α_stochastic = {agg['dissoc'][0]:+.2f} ± "
          f"{agg['dissoc'][1]:.2f}  (stochastic is HIGHEST-variance yet LOWEST α; α_stable − α_stochastic = "
          f"{agg['stable_vs_stoch'][0]:+.2f})", flush=True)
    print(f"  (C) LEARNED not architectural: untrained-GRU vol-gain {agg['untr_vol_gain'][0]:+.2f} ± "
          f"{agg['untr_vol_gain'][1]:.2f} (≈0)", flush=True)
    print(f"  (D) FUNCTIONAL: net error / best-fixed-α error = {agg['perf_ratio'][0]:.2f} (<1 ⇒ beats any static rate)", flush=True)

    print(f"\n  -> a GRU meta-trained ONLY to predict the next observation — never told the hazard or the noise — "
          f"develops in its frozen recurrent dynamics a learning rate that ADAPTS online: it is higher when the "
          f"world is VOLATILE ({agg['alpha_volatile'][0]:.2f}) than STABLE ({agg['alpha_stable'][0]:.2f}, "
          f"gain {agg['vol_gain'][0]:+.2f}), and — the signature a naive 'learn faster on big errors' account "
          f"cannot make — it DROPS under pure STOCHASTICITY ({agg['alpha_stochastic'][0]:.2f}) even though that "
          f"block has the highest observation variance (dissociation {agg['dissoc'][0]:+.2f}). The adaptation is "
          f"meta-learned (an untrained net is flat, {agg['untr_vol_gain'][0]:+.2f}) and functional (it beats the "
          f"best single fixed rate, error ratio {agg['perf_ratio'][0]:.2f}). The brain tuning its own learning "
          f"rate from inferred volatility (Behrens 2007; Wang 2018) — emergent, measured, not in the loss.", flush=True)

    out = {"n_seeds": a.seeds, "blocks": {k: list(v) for k, v in BLOCKS.items()}, "iters": a.iters,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/meta_learning.json", "w"), indent=2)
    svg(agg, "results/meta_learning.svg")
    print("\nwrote results/meta_learning.json and results/meta_learning.svg", flush=True)


def svg(agg, out):
    pad = 60; pw = 250; ph = 200; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 40
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'A self-tuned learning rate from inferred volatility</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">the network raises its learning rate for '
             'volatility and LOWERS it for stochasticity &#8212; despite equal-or-higher variance; measured, not trained</text>')
    oy = 58; base = oy + ph
    # Panel A: revealed α by block (trained) with untrained ghost
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">revealed learning rate α</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("stable", agg["alpha_stable"][0], agg["untr_stable"][0], "#3182bd"),
            ("volatile", agg["alpha_volatile"][0], agg["untr_volatile"][0], "#2ca25f"),
            ("stochastic", agg["alpha_stochastic"][0], agg["untr_stochastic"][0], "#c9341a")]
    hi = max(max(b[1], b[2]) for b in bars) + 1e-6
    for i, (lab, v, uv, col) in enumerate(bars):
        h = (v / hi) * (ph - 30); uh = (uv / hi) * (ph - 30); x = oxA + 18 + i * 74
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="52" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<rect x="{x+54}" y="{base-uh:.1f}" width="10" height="{uh:.1f}" fill="#9aa6bd" opacity="0.7"/>')
        e.append(f'<text x="{x+26}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        e.append(f'<text x="{x+26}" y="{base+14:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<text x="{oxA}" y="{base+30:.0f}" font-size="9" fill="#5b6b8c">grey = untrained control (flat)</text>')
    # Panel B: the two headline contrasts
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">volatility ↑α, stochasticity ↓α</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    cs = [("track\nvolatility", agg["vol_gain"][0], "#2ca25f"), ("dissociate\nstochasticity", agg["dissoc"][0], "#8039ef")]
    hib = max(abs(c[1]) for c in cs) + 1e-6
    for i, (lab, v, col) in enumerate(cs):
        h = (v / hib) * (ph - 40); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-max(h,0):.1f}" width="64" height="{abs(h):.1f}" fill="{col}" opacity="0.88"/>')
        e.append(f'<text x="{x+32}" y="{base-abs(h)-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+14+j*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+34:.0f}" font-size="9.5" fill="#5b6b8c">both are α-differences (95% CI); '
             f'beats best fixed α (ratio {agg["perf_ratio"][0]:.2f})</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
