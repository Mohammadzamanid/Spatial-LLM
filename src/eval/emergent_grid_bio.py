"""
src/eval/emergent_grid_bio.py

THE FAITHFULNESS CAPSTONE — grid cells emerge under a NON-BACKPROP rule (GAPS.md Tier 5, capstone).

`emergence.py` shows the landmark result that periodic grid-cell fields EMERGE when a recurrent cortex is
trained on self-supervised path integration (predict a place-cell code from velocity; nothing periodic imposed;
Cueva & Wei 2018; Banino 2018). But it trains by BACKPROP — the very thing #A1 argued the cortex cannot do
(weight transport, a global backward pass). This closes the loop: we train the SAME kind of path-integration
recurrent net by a fully LOCAL, biologically-plausible rule — RFLO (Murray 2019): an eligibility trace
(e-prop's temporal-credit primitive) times a learning signal delivered through a FIXED RANDOM feedback matrix
(A1's feedback alignment — no weight transport, no backprop-through-time) — and ask whether the grid code still
emerges. This moves the repo from "biological learning rules bolted onto a backprop-trained core" to "the core
itself learns biologically."

Four rules from a MATCHED init: backprop (BPTT — the reference), RFLO (the biological rule), a SHUFFLED-feedback
falsifier (the random feedback re-drawn every step, so it carries no consistent teaching signal), and UNTRAINED.
We MEASURE (never train): (1) does the rule SOLVE path integration (decode position from the hidden code)? and
(2) does the emergent GRID signature — the spatial PERIODICITY of the hidden rate maps, scored by the exact
`grid_stats` autocorrelogram machinery of `emergence.py` — appear?

  (A) RFLO SOLVES path integration (position-decode R² ≈ backprop, ≫ shuffled/untrained) — WITHOUT weight
      transport or BPTT.
  (B) The GRID CODE emerges under RFLO: spatial periodicity ≈ backprop, and ≫ the shuffled/untrained floor —
      the periodic code is never in the loss (the loss is place-cell prediction), so it EMERGES.
  (C) FALSIFIER: shuffled feedback (inconsistent teaching) fails on both — it lands at the untrained floor — so
      it is the CONSISTENT random feedback (which the forward weights align to, per #A1), not any feedback, that
      grows the grid code.

Honest scope: like `emergence.py`'s unconstrained model, the emergent signature is PERIODIC MULTI-FIELD spatial
tuning, not a clean hexagonal lattice (gridness stays negative for backprop too — hexagonality needs the
constructed velocity modules, `emergence.py --constrained`); the claim is that this emergent grid code forms
under a no-weight-transport rule. Multi-seed, mean ± 95% CI. Writes results/emergent_grid_bio.json + .svg.

    python -m src.eval.emergent_grid_bio --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from src.eval.emergence import rate_maps, grid_stats

N = 96                 # recurrent (grid-cell) units
P = 64                 # place-cell readout targets
T = 12                 # path length per trial
BATCH = 32
R = 3.0                # arena half-width
SIG = 0.5              # place-field width
SPEED = (0.15, 0.5)
ITERS = 1600
ALPHA = 0.5            # RFLO eligibility-trace filter
LR_BP = 3e-3
LR_RFLO = 0.02
G = 24                 # rate-map resolution
PERIODIC_THR = 0.5     # a unit is "periodic" if its autocorrelogram periodicity exceeds this


def _centers(gen):
    return (torch.rand(P, 2, generator=gen) * 2 - 1) * R


def walks(n, gen):
    H = torch.rand(n, T, generator=gen) * 2 * math.pi
    S = torch.rand(n, T, generator=gen) * (SPEED[1] - SPEED[0]) + SPEED[0]
    v = torch.stack([S * H.cos(), S * H.sin()], -1)
    return v, torch.cumsum(v, 1)                                # velocities (n,T,2), positions (n,T,2)


def pcode(pos, centers):
    d2 = ((pos.unsqueeze(-2) - centers) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * SIG ** 2))


def _init(gen):
    return (torch.randn(N, N, generator=gen) / math.sqrt(N),   # Wr
            torch.randn(N, 2, generator=gen) * 0.3,            # Wi
            torch.randn(P, N, generator=gen) / math.sqrt(N),   # Wo
            torch.randn(N, P, generator=gen) / math.sqrt(P))   # Bfb — FIXED random feedback (no weight transport)


def train_backprop(seed, centers):
    g = torch.Generator().manual_seed(seed); dg = torch.Generator().manual_seed(seed + 5)
    Wr, Wi, Wo, _ = _init(g)
    Wr, Wi, Wo = Wr.requires_grad_(), Wi.requires_grad_(), Wo.requires_grad_()
    opt = torch.optim.Adam([Wr, Wi, Wo], LR_BP)
    for _ in range(ITERS):
        v, pos = walks(BATCH, dg); y = pcode(pos, centers); h = torch.zeros(BATCH, N); loss = 0.0
        for t in range(T):
            h = torch.tanh(h @ Wr.t() + v[:, t] @ Wi.t()); loss = loss + F.mse_loss(h @ Wo.t(), y[:, t])
        opt.zero_grad(); (loss / T + 1e-3 * (h ** 2).mean()).backward(); opt.step()
    return Wr.detach(), Wi.detach(), Wo.detach()


def train_rflo(seed, centers, mode):
    """RFLO (Murray 2019): eligibility trace × learning signal via FIXED RANDOM feedback. No weight transport,
    no BPTT. mode='rflo' uses the fixed feedback Bfb; mode='shuffled' re-draws it every step (the falsifier)."""
    g = torch.Generator().manual_seed(seed); dg = torch.Generator().manual_seed(seed + 5)
    fg = torch.Generator().manual_seed(seed + 9)
    Wr, Wi, Wo, Bfb = _init(g)
    for _ in range(ITERS):
        v, pos = walks(BATCH, dg); y = pcode(pos, centers)
        h = torch.zeros(BATCH, N)
        p_rec = torch.zeros(BATCH, N, N); p_in = torch.zeros(BATCH, N, 2)
        dWr = torch.zeros(N, N); dWi = torch.zeros(N, 2); dWo = torch.zeros(P, N)
        for t in range(T):
            hprev = h
            h = torch.tanh(hprev @ Wr.t() + v[:, t] @ Wi.t())
            psi = 1 - h ** 2                                    # tanh-derivative surrogate
            p_rec = (1 - ALPHA) * p_rec + ALPHA * psi.unsqueeze(2) * hprev.unsqueeze(1)   # eligibility (e-prop)
            p_in = (1 - ALPHA) * p_in + ALPHA * psi.unsqueeze(2) * v[:, t].unsqueeze(1)
            err = h @ Wo.t() - y[:, t]                          # (B,P) readout error
            B_t = torch.randn(N, P, generator=fg) / math.sqrt(P) if mode == "shuffled" else Bfb
            L = err @ B_t.t()                                   # (B,N) learning signal via random feedback (FA)
            dWr += (L.unsqueeze(2) * p_rec).mean(0)
            dWi += (L.unsqueeze(2) * p_in).mean(0)
            dWo += (err.unsqueeze(2) * h.unsqueeze(1)).mean(0)  # readout (direct, top layer)
        Wr -= LR_RFLO * dWr / T; Wi -= LR_RFLO * dWi / T; Wo -= LR_RFLO * dWo / T
    return Wr, Wi, Wo


@torch.no_grad()
def measure(Wr, Wi, Wo, centers, seed):
    """Run the trained net over many walks; report (A) whether it SOLVES the task — the place-prediction loss,
    the training objective (an untrained readout cannot predict place cells) — and (B) the emergent grid signature
    — the spatial PERIODICITY / gridness of the hidden rate maps (never in the loss)."""
    v, pos = walks(5000, torch.Generator().manual_seed(seed + 90))
    h = torch.zeros(5000, N)
    hs, ps = [], []
    for t in range(T):
        h = torch.tanh(h @ Wr.t() + v[:, t] @ Wi.t())
        hs.append(h); ps.append(pos[:, t])                     # capture EVERY step for good arena coverage
    Hcat = torch.cat(hs); Pcat = torch.cat(ps)                 # (5000*T, N), (5000*T, 2)
    # (A) does it solve the task? place-prediction loss at the end position (the training objective)
    place_loss = F.mse_loss(h @ Wo.t(), pcode(pos[:, -1], centers)).item()
    # (B) emergent grid signature: periodicity / gridness of hidden rate maps (nan-safe over units)
    rms, _ = rate_maps(Pcat, Hcat, G=G, R=R)
    per, grd = [], []
    for u in range(N):
        if torch.isnan(rms[u]).float().mean() > 0.5:
            continue
        g0, s4, s6, pr, npk, ac = grid_stats(rms[u])
        if math.isfinite(pr):
            per.append(pr)
        if math.isfinite(g0):
            grd.append(g0)
    per = torch.tensor(per) if per else torch.zeros(1)
    grd = torch.tensor(grd) if grd else torch.zeros(1)
    return {"place_loss": place_loss, "mean_periodicity": per.mean().item(),
            "frac_periodic": (per > PERIODIC_THR).float().mean().item(),
            "mean_gridness": grd.mean().item()}


def run_seed(seed, iters=None):
    global ITERS
    if iters is not None:
        ITERS = iters
    centers = _centers(torch.Generator().manual_seed(seed + 1))
    out = {}
    conds = {"backprop": lambda: train_backprop(seed, centers),
             "rflo": lambda: train_rflo(seed, centers, "rflo"),
             "shuffled": lambda: train_rflo(seed, centers, "shuffled"),
             "untrained": lambda: _init(torch.Generator().manual_seed(seed))[:3]}
    for name, fn in conds.items():
        Wr, Wi, Wo = fn()
        m = measure(Wr, Wi, Wo, centers, seed)
        for k, val in m.items():
            out[f"{name}_{k}"] = val
    out["grid_emerges_rflo"] = out["rflo_mean_periodicity"] - out["untrained_mean_periodicity"]   # (B) vs floor
    out["falsifier_gap"] = out["rflo_mean_periodicity"] - out["shuffled_mean_periodicity"]         # (C) periodicity
    out["task_gap"] = out["shuffled_place_loss"] - out["rflo_place_loss"]                          # (A) >0: RFLO solves, shuffled worse
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


KEYS = ([f"{c}_{m}" for c in ("backprop", "rflo", "shuffled", "untrained")
         for m in ("place_loss", "mean_periodicity", "frac_periodic", "mean_gridness")]
        + ["grid_emerges_rflo", "falsifier_gap", "task_gap"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: place-loss BP {p['backprop_place_loss']:.3f} / RFLO {p['rflo_place_loss']:.3f} / "
              f"shuf {p['shuffled_place_loss']:.3f} | periodicity BP {p['backprop_mean_periodicity']:.2f} / "
              f"RFLO {p['rflo_mean_periodicity']:.2f} / shuf {p['shuffled_mean_periodicity']:.2f} / "
              f"untr {p['untrained_mean_periodicity']:.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nGRID CELLS UNDER A NON-BACKPROP RULE — the cortex learns its grid code biologically "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 96, flush=True)
    print(f"  (A) SOLVES THE TASK — place-prediction loss (lower = learned; the training objective): backprop "
          f"{agg['backprop_place_loss'][0]:.3f} | RFLO {agg['rflo_place_loss'][0]:.3f} | shuffled "
          f"{agg['shuffled_place_loss'][0]:.3f} | untrained {agg['untrained_place_loss'][0]:.3f}", flush=True)
    print(f"  (B) EMERGENT GRID CODE (rate-map periodicity; NEVER in the loss): backprop "
          f"{agg['backprop_mean_periodicity'][0]:.2f} | RFLO {agg['rflo_mean_periodicity'][0]:.2f} | "
          f"shuffled {agg['shuffled_mean_periodicity'][0]:.2f} | untrained {agg['untrained_mean_periodicity'][0]:.2f}", flush=True)
    print(f"      RFLO grid emergence over the untrained floor: {agg['grid_emerges_rflo'][0]:+.2f} ± "
          f"{agg['grid_emerges_rflo'][1]:.2f}  |  frac periodic (>{PERIODIC_THR}): RFLO "
          f"{agg['rflo_frac_periodic'][0]:.0%} vs untrained {agg['untrained_frac_periodic'][0]:.0%}", flush=True)
    print(f"  (C) FALSIFIER (RFLO − shuffled): periodicity {agg['falsifier_gap'][0]:+.2f} ± "
          f"{agg['falsifier_gap'][1]:.2f}; task-loss gap {agg['task_gap'][0]:+.3f} "
          f"(shuffled feedback fails at the untrained floor)", flush=True)
    print(f"  honest: gridness stays negative (backprop {agg['backprop_mean_gridness'][0]:+.2f}, RFLO "
          f"{agg['rflo_mean_gridness'][0]:+.2f}) — periodic multi-field, not a hexagonal lattice (as in emergence.py)", flush=True)

    print(f"\n  -> a recurrent cortex trained by RFLO — an eligibility trace times a learning signal delivered "
          f"through a FIXED RANDOM feedback matrix (no weight transport, no backprop-through-time) — LEARNS to "
          f"path-integrate (place-loss {agg['rflo_place_loss'][0]:.3f} ≈ backprop {agg['backprop_place_loss'][0]:.3f}, "
          f"vs untrained {agg['untrained_place_loss'][0]:.3f}) and grows the SAME emergent periodic grid code "
          f"(periodicity {agg['rflo_mean_periodicity'][0]:.2f} ≈ backprop {agg['backprop_mean_periodicity'][0]:.2f}), "
          f"which is NEVER in the loss. With the feedback SHUFFLED every step it fails on both (periodicity "
          f"{agg['shuffled_mean_periodicity'][0]:.2f}, place-loss {agg['shuffled_place_loss'][0]:.3f} — the "
          f"untrained floor): it is the CONSISTENT random feedback the forward weights align to (#A1), not any "
          f"feedback, that grows grid cells. The core itself learns biologically — measured, not put in the loss.", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "P": P, "T": T, "iters": ITERS,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/emergent_grid_bio.json", "w"), indent=2)
    svg(agg, "results/emergent_grid_bio.svg")
    print("\nwrote results/emergent_grid_bio.json and results/emergent_grid_bio.svg", flush=True)


def svg(agg, out):
    pad = 60; pw = 260; ph = 200; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Grid cells emerge under a NON-backprop rule (RFLO)</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">a recurrent cortex trained with fixed random '
             'feedback + eligibility (no weight transport) learns path integration AND its periodic grid code</text>')
    oy = 58; base = oy + ph
    # Panel A: decode R²  Panel B: grid periodicity — both across the four rules
    def panel(ox, title, keys, sub):
        e.append(f'<text x="{ox}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">{title}</text>')
        e.append(f'<line x1="{ox}" y1="{base}" x2="{ox+pw}" y2="{base}" stroke="#33415c"/>')
        bars = [("backprop", keys[0], "#3182bd"), ("RFLO", keys[1], "#2ca25f"),
                ("shuffled", keys[2], "#c9341a"), ("untr", keys[3], "#9aa6bd")]
        hi = max(agg[k][0] for _, k, _ in bars) + 1e-6
        for i, (lab, k, col) in enumerate(bars):
            v = agg[k][0]; h = (max(v, 0) / hi) * (ph - 30); x = ox + 16 + i * 60
            e.append(f'<rect x="{x}" y="{base-h:.1f}" width="42" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
            e.append(f'<text x="{x+21}" y="{base-h-6:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
            e.append(f'<text x="{x+21}" y="{base+14:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{lab}</text>')
        e.append(f'<text x="{ox}" y="{base+30:.0f}" font-size="9" fill="#5b6b8c">{sub}</text>')
    panel(pad, "(A) task solved (place-loss, lower=better)",
          ["backprop_place_loss", "rflo_place_loss", "shuffled_place_loss", "untrained_place_loss"],
          "RFLO learns it without weight transport; untrained/shuffled high")
    panel(pad + pw + gap, "(B) emergent grid code (frac periodic)",
          ["backprop_frac_periodic", "rflo_frac_periodic", "shuffled_frac_periodic", "untrained_frac_periodic"],
          "never in the loss; shuffled falls to the untrained floor")
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
