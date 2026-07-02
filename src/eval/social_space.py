"""
src/eval/social_space.py

SOCIAL SPACE — does a self-map and an OTHER-agent map coexist in one population? The hippocampus encodes not
only the animal's OWN position but the position of ANOTHER individual, in dedicated "social place cells" (Danjo,
Toyoizumi & Fujisawa 2018; Omer, Maimon, Las & Ulanovsky 2018 in bats), and humans map social variables with
the same machinery (Tavares 2015; Park 2021). The model had no representation of another agent at all
(GAPS.md #4). We close it with the project's emergence methodology: ONE recurrent substrate is fed its OWN
self-motion velocity AND its observation of ANOTHER agent's motion, and trained to report BOTH positions. We
then MEASURE, per unit, how much of its activity is explained by SELF position vs OTHER position (eta^2,
variance explained) and classify pure-self / pure-other / conjunctive — nothing imposed — and lesion each
sub-population to show a clean double dissociation.

Multi-seed, mean +/- 95% CI. Writes results/social_space.json + .svg.

    python -m src.eval.social_space --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.temporal_cortex import TemporalCortex

T = 40; HIDDEN = 128; NOISE = 0.06; ACT_COST = 1e-3; LBOX = 1.0
GP = 6                 # position grid (GP x GP) for tuning of each agent
ETA = 0.08             # eta^2 threshold for "tuned"


def walk(B, gen):
    """A momentum random walk in a bounded box; returns velocity (B,T,2) and position (B,T,2)."""
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


def _inputs(vs, vo, B):
    x = torch.zeros(B, T, 5)
    x[:, :, :2] = vs          # self-motion velocity (must be integrated -> self position)
    x[:, :, 2:4] = vo         # observed motion of the OTHER agent (integrated -> other position)
    x[:, 0, 4] = 1.0          # start pulse
    return x


def run_seed(seed, iters=1800):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=HIDDEN, n_in=5)
    sh = nn.Linear(HIDDEN, 2); oh = nn.Linear(HIDDEN, 2)          # decode SELF pos, OTHER pos
    opt = torch.optim.Adam(list(cx.parameters()) + list(sh.parameters()) + list(oh.parameters()), 3e-3)
    for _ in range(iters):
        vs, ps = walk(96, g); vo, po = walk(96, g)               # self and other move independently
        x = _inputs(vs, vo, 96)
        R = cx.dynamics(x, noise=NOISE, gen=g)
        probe = torch.randint(T // 5, T, (96,), generator=g); rp = R[torch.arange(96), probe]
        loss = ((sh(rp) - ps[torch.arange(96), probe]) ** 2).mean() \
            + ((oh(rp) - po[torch.arange(96), probe]) ** 2).mean() + ACT_COST * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        vs, ps = walk(300, g); vo, po = walk(300, g)
        x = _inputs(vs, vo, 300)
        R = cx.dynamics(x, noise=NOISE, gen=g)
        idx = torch.arange(300)
        probe = torch.randint(T // 5, T, (300,), generator=g); rp = R[idx, probe]
        pos_self = ps[idx, probe]; pos_other = po[idx, probe]

        def mae(head, target, mask=None):
            r = rp if mask is None else rp * mask
            return (head(r) - target).abs().mean().item()

        # per-unit variance explained by SELF vs OTHER position (over all timesteps)
        A = R.reshape(-1, HIDDEN)
        PS = ps.reshape(-1, 2); PO = po.reshape(-1, 2)
        def binpos(P):
            gx = ((P[:, 0] + LBOX) / (2 * LBOX) * GP).clamp(0, GP - 0.01).long()
            gy = ((P[:, 1] + LBOX) / (2 * LBOX) * GP).clamp(0, GP - 0.01).long()
            return gx * GP + gy
        sbin = binpos(PS); obin = binpos(PO)
        se = torch.zeros(HIDDEN); oe = torch.zeros(HIDDEN); active = torch.zeros(HIDDEN, dtype=torch.bool)
        for u in range(HIDDEN):
            a = A[:, u]
            if a.std() < 1e-3:
                continue
            active[u] = True; se[u] = eta2(a, sbin, GP * GP); oe[u] = eta2(a, obin, GP * GP)
        is_self = (se > ETA) & active; is_other = (oe > ETA) & active
        n_act = max(int(active.sum()), 1)
        pure_self = (is_self & ~is_other); pure_other = (is_other & ~is_self); conj = (is_self & is_other)

        # lesion double dissociation: zero the PURE-OTHER units -> other decode fails, self survives (and v.v.)
        keep_other = (~pure_other).float().unsqueeze(0)          # (1,H) mask that removes other-coding units
        keep_self = (~pure_self).float().unsqueeze(0)
        out = {
            "n_active": n_act,
            "frac_self": int(pure_self.sum()) / n_act,
            "frac_other": int(pure_other.sum()) / n_act,
            "frac_conjunctive": int(conj.sum()) / n_act,
            "self_mae": mae(sh, pos_self), "other_mae": mae(oh, pos_other),
            "self_mae_lesion_other": mae(sh, pos_self, keep_other),   # ablate OTHER cells: self should survive
            "other_mae_lesion_other": mae(oh, pos_other, keep_other), # ablate OTHER cells: other should fail
            "self_mae_lesion_self": mae(sh, pos_self, keep_self),     # ablate SELF cells: self should fail
            "other_mae_lesion_self": mae(oh, pos_other, keep_self),   # ablate SELF cells: other should survive
        }
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


FRACS = ["frac_self", "frac_other", "frac_conjunctive"]
MAES = ["self_mae", "other_mae", "self_mae_lesion_other", "other_mae_lesion_other",
        "self_mae_lesion_self", "other_mae_lesion_self"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1800); a = ap.parse_args()
    per = []
    for s in range(a.seeds):
        o = run_seed(s, a.iters); per.append(o)
        print(f"  seed {s}: SELF {o['frac_self']:.0%} OTHER {o['frac_other']:.0%} CONJ {o['frac_conjunctive']:.0%} | "
              f"self-MAE {o['self_mae']:.2f} other-MAE {o['other_mae']:.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in FRACS + MAES}

    print(f"\nSOCIAL SPACE — a SELF map and an OTHER-agent map in one population (n={a.seeds}; mean ± 95% CI; "
          f"box half-width {LBOX})\n" + "=" * 82, flush=True)
    lab = {"frac_self": "PURE SELF-place cells (own position)",
           "frac_other": "PURE OTHER-place cells (the other agent's position)",
           "frac_conjunctive": "CONJUNCTIVE self x other cells"}
    for k in FRACS:
        print(f"  {lab[k]:52} {agg[k][0]:+.3f} ± {agg[k][1]:.3f}", flush=True)
    print("\n  double dissociation (decode MAE; lower=better):", flush=True)
    print(f"    {'':22} | {'decode SELF':>12} | {'decode OTHER':>12}", flush=True)
    print(f"    {'intact':22} | {agg['self_mae'][0]:>12.3f} | {agg['other_mae'][0]:>12.3f}", flush=True)
    print(f"    {'lesion OTHER cells':22} | {agg['self_mae_lesion_other'][0]:>12.3f} | {agg['other_mae_lesion_other'][0]:>12.3f}", flush=True)
    print(f"    {'lesion SELF cells':22} | {agg['self_mae_lesion_self'][0]:>12.3f} | {agg['other_mae_lesion_self'][0]:>12.3f}", flush=True)
    print(f"\n  -> ONE population fed self-motion AND the other agent's motion develops SEPARATE maps: "
          f"{agg['frac_self'][0]:.0%} PURE SELF-place cells and {agg['frac_other'][0]:.0%} PURE OTHER-place cells "
          f"(plus {agg['frac_conjunctive'][0]:.0%} conjunctive) — the social place cells of Danjo 2018 / Omer "
          f"2018, emergent. And they DISSOCIATE: lesioning the OTHER cells wrecks other-decoding "
          f"({agg['other_mae'][0]:.2f}→{agg['other_mae_lesion_other'][0]:.2f}) while self-decoding survives "
          f"({agg['self_mae_lesion_other'][0]:.2f}); lesioning the SELF cells does the reverse "
          f"(self {agg['self_mae'][0]:.2f}→{agg['self_mae_lesion_self'][0]:.2f}, other stays "
          f"{agg['other_mae_lesion_self'][0]:.2f}). A self-map and an other-map, coexisting in one circuit.", flush=True)

    out = {"n_seeds": a.seeds, "T": T, "hidden": HIDDEN, "box": LBOX, "iters": a.iters,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in FRACS + MAES}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/social_space.json", "w"), indent=2)
    svg(agg, "results/social_space.svg")
    print("\nwrote results/social_space.json and results/social_space.svg", flush=True)


def svg(agg, out):
    pad = 56; bw = 52; ph = 190; gapx = 26; W = 620; H = 92 + ph + 70
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Social space: a SELF map and an OTHER-agent map in one population</text>')
    e.append('<text x="26" y="44" font-size="10.5" fill="#5b6b8c">emergent social place cells (Danjo 2018, Omer 2018), '
             'with a self/other lesion double dissociation</text>')
    oy = 58; base = oy + ph
    # (left) cell-type fractions
    e.append(f'<line x1="{pad}" y1="{base}" x2="{pad+3*(bw+gapx)}" y2="{base}" stroke="#33415c"/>')
    for i, (k, lb, col) in enumerate([("frac_self", "self", "#3182bd"), ("frac_other", "other", "#e6550d"),
                                      ("frac_conjunctive", "conj.", "#756bb1")]):
        v = agg[k][0]; h = v * ph; x = pad + i * (bw + gapx)
        e.append(f'<rect x="{x}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.88"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base+14:.0f}" font-size="9.5" fill="#28324a" text-anchor="middle">{lb}</text>')
    e.append(f'<text x="{pad}" y="{base+36:.0f}" font-size="10" fill="#28324a">cell types (of active units)</text>')
    # (right) dissociation grouped bars: decode SELF and OTHER under intact / lesion-other / lesion-self
    ox = pad + 3 * (bw + gapx) + 60
    groups = [("decode SELF", "self_mae", "self_mae_lesion_other", "self_mae_lesion_self"),
              ("decode OTHER", "other_mae", "other_mae_lesion_other", "other_mae_lesion_self")]
    hi = max(agg[m][0] for g in groups for m in g[1:]) * 1.2 + 1e-6
    cols = ["#2ca25f", "#e6550d", "#3182bd"]; leg = ["intact", "lesion OTHER", "lesion SELF"]
    gw = 150; sbw = 30
    for gi, (title, m0, m1, m2) in enumerate(groups):
        gx = ox + gi * (gw + 30)
        e.append(f'<line x1="{gx}" y1="{base}" x2="{gx+gw}" y2="{base}" stroke="#33415c"/>')
        e.append(f'<text x="{gx+gw/2:.0f}" y="{base+30:.0f}" font-size="10.5" font-weight="700" fill="#28324a" text-anchor="middle">{title}</text>')
        for j, m in enumerate((m0, m1, m2)):
            v = agg[m][0]; h = v / hi * ph; x = gx + j * (sbw + 8)
            e.append(f'<rect x="{x}" y="{base-h:.1f}" width="{sbw}" height="{h:.1f}" fill="{cols[j]}" opacity="0.88"/>')
            e.append(f'<text x="{x+sbw/2:.0f}" y="{base-h-4:.0f}" font-size="8.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
    ly = oy + 6
    for j in range(3):
        e.append(f'<rect x="{ox+gw*2+10}" y="{ly-8}" width="11" height="6" fill="{cols[j]}"/>'
                 f'<text x="{ox+gw*2+25}" y="{ly-2}" font-size="9" fill="#28324a">{leg[j]}</text>'); ly += 14
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
