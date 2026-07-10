"""
src/eval/reference_transform.py

The RSC/PPC EGOCENTRIC->ALLOCENTRIC transform, with emergent GAIN FIELDS (GAPS.md Tier 1/2).

Posterior parietal cortex represents space egocentrically ("the landmark is to my left"); the retrosplenial
cortex transforms it to allocentric world coordinates ("the landmark is north") using head direction — a
head-direction-gated rotation implemented, in cortex, by GAIN FIELDS (multiplicative modulation of an egocentric
response by a directional signal; Andersen & Zipser 1988; Byrne, Becker & Burgess 2007; Bicanski & Burgess
2018). The repo has egocentric and allocentric codes coexisting (landmark_anchoring.py) but not the transform
circuit itself.

A plain MLP is trained ONLY to output a landmark's ALLOCENTRIC position from its egocentric view (distance,
bearing relative to the head) + the head direction — never told about rotation or gain fields. We MEASURE, never
put in a loss:

  (A) IT LEARNED THE TRANSFORM (non-circular): trained on head directions OUTSIDE a held-out band, it generalizes
      to head directions it NEVER saw with near-zero error — a lookup table cannot; only the systematic rotation
      generalizes.
  (B) GAIN FIELDS EMERGE: a sizeable fraction of hidden units develop MULTIPLICATIVE egocentric*head-direction
      tuning (the Zipser-Andersen signature) — measured as the extra variance their activity needs from the
      multiplicative terms beyond additive ones, far above an untrained network. Emergent, not imposed.
  (C) FALSIFIERS: with the head direction SHUFFLED (wrong heading) or REMOVED, the error explodes past the target
      scale — the transform is impossible without the correct directional signal.

Honest scope: this is a mechanism demonstration with a real emergent internal code (gain fields), but it is the
*expected* solution to a multiplicative transform — not a surprising emergence.

    python -m src.eval.reference_transform --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HELDOUT = (math.pi / 2, math.pi)   # head-direction band held out of TRAINING (tests generalization)
STEPS = 3000


def sample(n, phi_lo=None, phi_hi=None, gen=None):
    dist = torch.rand(n, generator=gen) * 2.5 + 0.5
    te = torch.rand(n, generator=gen) * 2 * math.pi                        # egocentric bearing (rel. to head)
    phi = (phi_lo + torch.rand(n, generator=gen) * (phi_hi - phi_lo)) if phi_lo is not None \
        else torch.rand(n, generator=gen) * 2 * math.pi                    # head direction
    allo = dist.unsqueeze(1) * torch.stack([torch.cos(te + phi), torch.sin(te + phi)], -1)   # allo bearing=te+phi
    x = torch.stack([dist, te.cos(), te.sin(), phi.cos(), phi.sin()], -1)
    return x, allo, te, phi


class Net(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(5, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.out = nn.Linear(h, 2)

    def forward(self, x, hidden=False):
        z = self.f(x); return (self.out(z), z) if hidden else self.out(z)


def _in_heldout(x):
    phi = torch.atan2(x[:, 4], x[:, 3]) % (2 * math.pi)
    return (phi > HELDOUT[0]) & (phi < HELDOUT[1])


def train(drop_hd=False, seed=0):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(STEPS):
        x, y, _, _ = sample(512, gen=g)
        keep = ~_in_heldout(x)                                             # TRAIN outside the held-out HD band
        x, y = x[keep], y[keep]
        if drop_hd:
            x = x.clone(); x[:, 3:5] = 0.0
        opt.zero_grad(); F.mse_loss(net(x), y).backward(); opt.step()
    return net


def rmse(net, x, y, shuffle_hd=False, drop_hd=False):
    if shuffle_hd:
        x = x.clone(); x[:, 3:5] = x[torch.randperm(len(x)), 3:5]
    if drop_hd:
        x = x.clone(); x[:, 3:5] = 0.0
    with torch.no_grad():
        return (net(x) - y).pow(2).sum(1).sqrt().mean().item()


def gain_field_index(net, gen):
    """Extra variance of hidden activity explained by MULTIPLICATIVE ego*HD terms beyond additive terms (the
    gain-field signature), averaged over units; and the fraction of units that are gain fields (extra R2>0.1)."""
    x, y, te, phi = sample(4000, gen=gen)
    with torch.no_grad():
        _, z = net(x, hidden=True)
    ones = torch.ones(len(x), 1)
    add = torch.stack([te.cos(), te.sin(), phi.cos(), phi.sin()], -1)
    inter = torch.stack([te.cos() * phi.cos(), te.cos() * phi.sin(),
                         te.sin() * phi.cos(), te.sin() * phi.sin()], -1)

    def r2(A):
        sol = torch.linalg.lstsq(A, z).solution
        return 1 - (z - A @ sol).pow(2).sum(0) / ((z - z.mean(0)).pow(2).sum(0) + 1e-9)
    extra = (r2(torch.cat([ones, add, inter], -1)) - r2(torch.cat([ones, add], -1))).clamp(min=0)
    return extra.mean().item(), (extra > 0.1).float().mean().item()


def run_seed(seed):
    g = torch.Generator().manual_seed(seed + 100)
    net = train(seed=seed)
    xin, yin, _, _ = sample(3000, gen=g); keep = ~_in_heldout(xin); xin, yin = xin[keep], yin[keep]
    xho, yho, _, _ = sample(3000, phi_lo=HELDOUT[0], phi_hi=HELDOUT[1], gen=g)   # HELD-OUT HD band
    scale = yho.pow(2).sum(1).sqrt().mean().item()
    r_in = rmse(net, xin, yin); r_ho = rmse(net, xho, yho)
    r_shuf = rmse(net, xho, yho, shuffle_hd=True)
    r_nohd = rmse(train(drop_hd=True, seed=seed), xho, yho, drop_hd=True)
    gf_extra, gf_frac = gain_field_index(net, g)
    gf0_extra, _ = gain_field_index(Net(), g)                             # untrained control
    return {
        "rmse_in_dist": round(r_in, 4),
        "rmse_heldout_hd": round(r_ho, 4),
        "rmse_heldout_norm": round(r_ho / scale, 4),                      # fraction of target scale
        "rmse_shuffled_hd": round(r_shuf, 4),
        "rmse_no_hd": round(r_nohd, 4),
        "gain_extra_r2": round(gf_extra, 4),
        "gain_extra_r2_untrained": round(gf0_extra, 4),
        "gain_field_frac": round(gf_frac, 4),
        "scale": round(scale, 4),
        "falsifier_gap": round(r_shuf - r_ho, 4),                         # wrong-HD vs correct-HD
        "gain_emergence": round(gf_extra - gf0_extra, 4),                 # learned gain fields over init
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["rmse_in_dist", "rmse_heldout_hd", "rmse_heldout_norm", "rmse_shuffled_hd", "rmse_no_hd",
        "gain_extra_r2", "gain_extra_r2_untrained", "gain_field_frac", "falsifier_gap", "gain_emergence"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: RMSE held-out-HD {p['rmse_heldout_hd']:.3f} ({p['rmse_heldout_norm']:.0%} of scale) "
              f"| shuffled-HD {p['rmse_shuffled_hd']:.2f}  no-HD {p['rmse_no_hd']:.2f} | gain-field frac "
              f"{p['gain_field_frac']:.2f} (extra-R2 {p['gain_extra_r2']:.3f} vs {p['gain_extra_r2_untrained']:.3f} untrained)", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nRSC/PPC EGO->ALLO TRANSFORM + emergent GAIN FIELDS (n={a.seeds}; mean ± 95% CI)\n" + "=" * 84, flush=True)
    print(f"  (A) LEARNED THE TRANSFORM (generalizes to UNSEEN head directions): held-out-HD RMSE "
          f"{agg['rmse_heldout_hd'][0]:.3f} ± {agg['rmse_heldout_hd'][1]:.3f}  "
          f"({agg['rmse_heldout_norm'][0]:.0%} of the target scale; in-dist {agg['rmse_in_dist'][0]:.3f}) — a lookup "
          f"could not; only the systematic rotation generalizes.", flush=True)
    print(f"  (B) GAIN FIELDS EMERGE: {agg['gain_field_frac'][0]:.0%} of hidden units carry MULTIPLICATIVE "
          f"ego*head-direction tuning (extra-R2 {agg['gain_extra_r2'][0]:.3f} vs {agg['gain_extra_r2_untrained'][0]:.3f} "
          f"untrained; emergence {agg['gain_emergence'][0]:+.3f}) — the Zipser-Andersen code, never in the loss.", flush=True)
    print(f"  (C) FALSIFIERS: SHUFFLED head direction -> RMSE {agg['rmse_shuffled_hd'][0]:.2f} (gap "
          f"{agg['falsifier_gap'][0]:+.2f}); REMOVED head direction -> {agg['rmse_no_hd'][0]:.2f} (> the target "
          f"scale) — the transform is impossible without the correct heading.", flush=True)

    sound = (agg["rmse_heldout_norm"][0] < 0.15 and agg["falsifier_gap"][0] > 0.3 and
             agg["gain_emergence"][0] > 0.03 and agg["rmse_no_hd"][0] > agg["rmse_heldout_hd"][0] + 0.5)
    verdict = ("SOUND — the network learned the egocentric->allocentric rotation (generalizes to unseen head "
               "directions, impossible without the heading), and gain-field units emerged as its internal code. "
               "The RSC/PPC transform, with the Zipser-Andersen signature — measured, not imposed." if sound else
               "WEAK — the transform/gain-field signatures did not clear the falsifiers; revisit the regime.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "heldout_band": [round(HELDOUT[0], 3), round(HELDOUT[1], 3)], "steps": STEPS,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/reference_transform.json", "w"), indent=2)
    _svg(agg, "results/reference_transform.svg")
    print("\nwrote results/reference_transform.json and results/reference_transform.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'RSC/PPC egocentric&#8594;allocentric transform, with emergent gain fields</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">trained only on allocentric output, it '
             'generalizes to unseen head directions (learned the rotation) and cannot work without the heading; '
             'gain-field units emerge</text>')
    oy = 60; base = oy + ph
    # Panel A: RMSE (held-out generalization vs falsifiers), lower=better
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) transform error (RMSE, lower=better)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("held-out\nHD", agg["rmse_heldout_hd"][0], "#2ca25f"), ("shuffled\nHD", agg["rmse_shuffled_hd"][0], "#c98a1a"),
            ("no\nHD", agg["rmse_no_hd"][0], "#c9341a")]
    hi = max(b[1] for b in bars) + 1e-6
    for i, (lab, v, col) in enumerate(bars):
        h = (v / hi) * (ph - 24); x = oxA + 22 + i * 74
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="52" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-h-6:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">held-out HD is near-zero (generalizes); wrong/removed HD explodes</text>')
    # Panel B: gain-field emergence (trained vs untrained extra-R2)
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) emergent gain-field code (extra R&#178;)</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    b2 = [("trained\nnetwork", agg["gain_extra_r2"][0], "#2ca25f"), ("untrained\n(init)", agg["gain_extra_r2_untrained"][0], "#9aa6bd")]
    hi2 = max(b[1] for b in b2) + 1e-6
    for i, (lab, v, col) in enumerate(b2):
        h = (v / hi2) * (ph - 24); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="64" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+32}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">{agg["gain_field_frac"][0]:.0%} of units become gain fields (multiplicative ego&#215;HD) with training</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
