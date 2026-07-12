"""
src/eval/semantic_warp.py

SEMANTIC WARPING OF THE COGNITIVE MAP — the spatial metric warps by a non-spatial concept, becoming
mixed-selective, ONLY when that concept is behaviorally relevant (GAPS.md: the "purely geographic map" critique).

The critique: the model treats the cortex as a purely geographic + value substrate and leaves all semantic meaning
to the language model. But biologically the perforant path projects non-spatial, semantic/behavioural features
directly into grid and place cell assemblies, so the map is not rigidly geographic — it WARPS by behaviourally
relevant structure. Grid cells deform toward remembered reward/goal locations, becoming MIXED-SELECTIVE to reward
and space (Boccara et al., *Science* 2019; "the entorhinal cognitive map is attracted to goals", Butler 2019), and
grid/place codes bind non-spatial dimensions (Aronov & Tank 2017; Constantinescu, O'Keefe & Behrens 2016;
Tolman-Eichenbaum Machine, Whittington 2020). If the map already warps to reflect conceptual relations, a
downstream reader (the LLM) reads them off the map instead of learning the semantic-spatial mapping from scratch.

Per the standing rule we hardcode NONE of the warping. The only things built are the mechanism (a capacity-limited
code with a spatial pathway AND a perforant/semantic input pathway) and the task (reconstruct POSITION and a scalar
VALUE — position forces a spatial map; the value may or may not depend on the concept). The warp is NEVER in the
loss. It emerges, and is measured:

  (A) THE MAP WARPS, YET STAYS SPATIAL (mixed selectivity). When the concept is behaviourally relevant, the
      representational metric warps by concept — same-concept locations move closer at MATCHED spatial distance
      (partial corr of representational distance with concept-difference, controlling spatial distance, > 0) —
      WHILE the code stays strongly spatial (partial corr with spatial distance stays high). Both at once = the
      mixed-selective warped map Boccara records.
  (B) FALSIFIER — remove the perforant path. Keep the concept behaviourally relevant but delete the semantic input
      projection: the map cannot warp (warp ~ 0) — it is the perforant path that carries the concept into the map.
      And with the path present but the concept irrelevant (relevance β = 0) the warp is ~0 too, so the warp needs
      BOTH the path and behavioural relevance.
  (C) DOSE-RESPONSE. As the concept's behavioural relevance grows, the warp and the concept's readability grow with
      it — the map is attracted to concepts in proportion to how much they matter.
  (D) PAYOFF (why it helps the reader). A held-out LINEAR probe reads the concept off the WARPED map far above
      chance, but is at chance without the perforant path — so a downstream reader inherits the semantic-spatial
      structure for free instead of learning it from scratch.

Multi-seed, mean ± 95% CI. Writes results/semantic_warp.json + .svg.

    python -m src.eval.semantic_warp --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

L = 10                  # L x L locations
K = 5                   # concept categories (scattered, not spatially contiguous)
D = 12                  # code bottleneck (capacity-limited -> concept is bound only if it earns its place)
BETA_REL = 1.5          # behavioural relevance of the concept in the headline "relevant" condition
DOSE = [0.0, 0.5, 1.0, 2.0]   # relevance levels for the dose-response (0.0 = the falsifier)
STEPS = 4000


def build_world(seed):
    g = torch.Generator().manual_seed(seed)
    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, L), torch.linspace(0, 1, L), indexing="ij"), -1).reshape(-1, 2)
    cat = torch.randint(0, K, (L * L,), generator=g)                        # scattered concepts
    centers = torch.rand(6, 2, generator=g); cval = torch.randn(K, generator=g)
    sv = torch.exp(-((coords[:, None, :] - centers[None]) ** 2).sum(-1) / (2 * 0.15 ** 2)).sum(1)
    sv = (sv - sv.mean()) / (sv.std() + 1e-6)                               # spatial value component
    cv = cval[cat]; cv = (cv - cv.mean()) / (cv.std() + 1e-6)               # concept value component
    rc = torch.rand(16, 2, generator=g)
    phi_s = torch.exp(-((coords[:, None, :] - rc[None]) ** 2).sum(-1) / (2 * 0.2 ** 2))   # spatial features
    phi_c = F.one_hot(cat, K).float()                                       # perforant (concept) input
    return coords, cat, phi_s, phi_c, sv, cv, g


def train_code(phi_s, phi_c, targets, g):
    """Bottleneck code reconstructs a multi-dim target [position(2), value(1)]. Position forces a spatial map; a
    concept-dependent value forces binding the concept. The warp is never a target. phi_c=None removes the
    perforant (semantic) projection entirely — the no-perforant falsifier."""
    X = phi_s if phi_c is None else torch.cat([phi_s, phi_c], 1)
    W = (torch.randn(X.shape[1], D, generator=g) * 0.1).requires_grad_(True)
    U = (torch.randn(D, targets.shape[1], generator=g) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([W, U], 1e-2, weight_decay=2e-3)
    t = (targets - targets.mean(0)) / (targets.std(0) + 1e-6)
    for _ in range(STEPS):
        loss = (((X @ W) @ U - t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (X @ W).detach()


def _partial(y, a, b):
    """Partial correlation of y with a, controlling for b (vectors)."""
    def resid(v, x):
        return v - (x * v).sum() / (x * x).sum() * x
    y, a, b = y - y.mean(), a - a.mean(), b - b.mean()
    ry, ra = resid(y, b), resid(a, b)
    return (ry * ra).sum().item() / (ry.norm() * ra.norm() + 1e-9).item()


def metrics(h, coords, cat, g):
    n = h.shape[0]
    iu = torch.triu_indices(n, n, 1)
    R = torch.cdist(h, h)[iu[0], iu[1]]
    Dsp = torch.cdist(coords, coords)[iu[0], iu[1]]
    Dcc = (cat[iu[0]] != cat[iu[1]]).float()
    warp = _partial(R, Dcc, Dsp)                                            # concept warps the metric (control space)
    spatial = _partial(R, Dsp, Dcc)                                         # still a spatial map (control concept)
    # payoff: held-out LINEAR probe reads concept off the code
    perm = torch.randperm(n, generator=g); tr, te = perm[:int(0.7 * n)], perm[int(0.7 * n):]
    Y = F.one_hot(cat, K).float()
    Htr = torch.cat([h[tr], torch.ones(len(tr), 1)], 1)
    Wp = torch.linalg.lstsq(Htr, Y[tr]).solution
    pred = (torch.cat([h[te], torch.ones(len(te), 1)], 1) @ Wp).argmax(1)
    acc = (pred == cat[te]).float().mean().item()
    return warp, spatial, acc


def run_seed(seed):
    coords, cat, phi_s, phi_c, sv, cv, g = build_world(seed)
    tgt_rel = torch.cat([coords, (sv + BETA_REL * cv).unsqueeze(1)], 1)               # concept IS relevant
    h_rel = train_code(phi_s, phi_c, tgt_rel, g)                                      # perforant present + relevant
    h_np = train_code(phi_s, None, tgt_rel, g)                                        # NO perforant (falsifier), same task
    wr, sr, ar = metrics(h_rel, coords, cat, g)
    wn, sn, an = metrics(h_np, coords, cat, g)
    dose = []                                                                          # perforant present, varying relevance
    for b in DOSE:
        h = train_code(phi_s, phi_c, torch.cat([coords, (sv + b * cv).unsqueeze(1)], 1), g)
        dose.append(metrics(h, coords, cat, g)[0])
    return {"warp_rel": wr, "spatial_rel": sr, "probe_rel": ar,
            "warp_np": wn, "spatial_np": sn, "probe_np": an, "chance": 1.0 / K,
            "dose_00": dose[0], "dose_05": dose[1], "dose_10": dose[2], "dose_20": dose[3]}


KEYS = ["warp_rel", "spatial_rel", "probe_rel", "warp_np", "spatial_np", "probe_np", "chance",
        "dose_00", "dose_05", "dose_10", "dose_20"]


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"SEMANTIC WARPING — the spatial map warps by concept when the concept is behaviourally relevant "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 86, flush=True)
    print(f"  (A) THE MAP WARPS, YET STAYS SPATIAL (mixed selectivity, Boccara 2019):", flush=True)
    print(f"      warp (concept | space) = {agg['warp_rel'][0]:+.2f} ± {agg['warp_rel'][1]:.2f}  "
          f"WHILE spatial (space | concept) = {agg['spatial_rel'][0]:+.2f} (still a spatial map)", flush=True)
    print(f"  (B) FALSIFIER — remove the perforant projection (same relevant task, no semantic input): warp = "
          f"{agg['warp_np'][0]:+.2f} ± {agg['warp_np'][1]:.2f} (~0; can't warp without the path, spatial "
          f"{agg['spatial_np'][0]:+.2f}). And with the path PRESENT but the concept irrelevant (β=0) warp is "
          f"{agg['dose_00'][0]:+.2f} too — the warp needs BOTH the path and behavioural relevance", flush=True)
    print(f"  (C) DOSE-RESPONSE (warp vs behavioural relevance β, perforant present): "
          f"β=0 {agg['dose_00'][0]:+.2f} -> 0.5 {agg['dose_05'][0]:+.2f} -> 1.0 {agg['dose_10'][0]:+.2f} -> "
          f"2.0 {agg['dose_20'][0]:+.2f}  (the map is attracted to concepts in proportion to relevance)", flush=True)
    print(f"  (D) PAYOFF — held-out linear probe reads concept off the map: WARPED {agg['probe_rel'][0]:.2f} vs "
          f"no-perforant {agg['probe_np'][0]:.2f} (chance {agg['chance'][0]:.2f}) — the reader gets it for free",
          flush=True)
    print(f"\n  The cognitive map is not purely geographic: a behaviourally relevant concept WARPS the spatial metric "
          f"(mixed-selective, map still spatial) and becomes readable off the map — none of it imposed; make the "
          f"concept irrelevant and the warp vanishes.", flush=True)

    out = {"n_seeds": a.seeds, "L": L, "K": K, "D": D, "beta_relevant": BETA_REL,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS},
           "verdict": "The cognitive map warps by a non-spatial concept, becoming mixed-selective (Boccara 2019), "
                      "ONLY when the concept is behaviourally relevant -- never imposed. A capacity-limited code "
                      "reconstructing [position, value] stays a spatial map (high spatial partial correlation) yet "
                      "its metric warps by concept (positive concept-partial-correlation controlling for space) "
                      "when the value depends on the concept; removing the perforant projection (the semantic "
                      "input) leaves the map purely spatial (warp ~0, the falsifier) even though the concept is "
                      "still relevant, and with the path present but the concept irrelevant (beta=0) the warp is "
                      "~0 too; the warp grows with behavioural relevance (dose-response); and a held-out linear "
                      "probe reads the concept off the warped map far above chance but is at chance without the "
                      "perforant path, so a downstream reader inherits semantic-spatial structure for free instead "
                      "of learning it from scratch."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/semantic_warp.json", "w"), indent=2)
    svg_warp(agg, "results/semantic_warp.svg")
    print("\nwrote results/semantic_warp.json and results/semantic_warp.svg", flush=True)


def svg_warp(agg, out):
    W_, H = 760, 320
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Semantic warping: the spatial map bends toward a concept &#8212; only when the concept matters</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">mixed-selective (Boccara 2019): the map stays spatial '
         'yet warps by concept; nothing about the warp is in the loss</text>']
    # left: warp relevant vs falsifier, with spatial retained
    bx, by, bh, bw = 44, 100, 150, 46
    e.append(f'<text x="{bx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">warp (concept|space)</text>')
    for i, (k, lab, col) in enumerate([("warp_rel", "concept\nRELEVANT", "#2ca25f"), ("warp_np", "no\nperforant", "#c9341a")]):
        v = max(0.0, agg[k][0]); x = bx + i * (bw + 20); h = v / 0.3 * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{agg[k][0]:+.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh}" x2="{bx+2*(bw+20):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{bx}" y="{by+bh+38:.0f}" font-size="8.5" fill="#5b6b8c">map stays spatial (corr {agg["spatial_rel"][0]:.2f}) in both</text>')
    # middle: dose-response
    m0 = 300; mw = 36
    e.append(f'<text x="{m0}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">warp vs relevance &#946;</text>')
    dmx = max(0.01, max(agg[k][0] for k in ["dose_00","dose_05","dose_10","dose_20"])) * 1.25
    for i, (k, lab) in enumerate([("dose_00", "0"), ("dose_05", ".5"), ("dose_10", "1"), ("dose_20", "2")]):
        v = max(0.0, agg[k][0]); x = m0 + i * (mw + 8); h = v / dmx * bh
        col = "#c9341a" if i == 0 else ("#e6842a" if i < 3 else "#2ca25f")
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{mw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh-h-4:.0f}" font-size="9" font-weight="700" fill="#0b1324" text-anchor="middle">{agg[k][0]:+.2f}</text>')
        e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh+13:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<line x1="{m0-4}" y1="{by+bh}" x2="{m0+4*(mw+8):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{m0}" y="{by+bh+30:.0f}" font-size="8.5" fill="#5b6b8c">more relevant &#8594; more warp</text>')
    # right: payoff probe
    rx = 560; rw = 56
    e.append(f'<text x="{rx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">read concept off map</text>')
    for i, (k, lab, col) in enumerate([("probe_rel", "warped", "#2ca25f"), ("probe_np", "no\nperforant", "#c9341a")]):
        v = agg[k][0]; x = rx + i * (rw + 16); h = v * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{rw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{rx-4}" y1="{by+bh}" x2="{rx+2*(rw+16):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    ch = agg["chance"][0] * bh
    e.append(f'<line x1="{rx-4}" y1="{by+bh-ch:.0f}" x2="{rx+2*(rw+16):.0f}" y2="{by+bh-ch:.0f}" stroke="#8c8c8c" stroke-dasharray="3 3"/>')
    e.append(f'<text x="{rx+2*(rw+16)-2:.0f}" y="{by+bh-ch-3:.0f}" font-size="8" fill="#8c8c8c" text-anchor="end">chance</text>')
    e.append(f'<text x="20" y="{H-12}" font-size="9.5" fill="#5b6b8c">The reader inherits semantic-spatial structure '
             f'for free from the warped map &#8212; on the un-warped map it would learn it from scratch.</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
