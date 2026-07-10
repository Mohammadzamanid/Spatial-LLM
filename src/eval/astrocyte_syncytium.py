"""
src/eval/astrocyte_syncytium.py

The astrocyte SYNCYTIUM — spatial-density-gated plasticity + heterosynaptic binding (GAPS.md Tier 2).

The repo already has a POINT-WISE astrocyte organ (`astrocyte_plasticity.py`, #B4): a slow glial gate that
throttles each synapse by ITS OWN activity. Astrocytes are also coupled into a gap-junction SYNCYTIUM across
which Ca2+ SPREADS (Scemes & Giaume 2006; Cornell-Bell 1990; the substrate for Ca2+ waves) — letting one
synapse's astrocyte influence its NEIGHBOURS. We ask, honestly, what that spatial coupling computes that the
point-wise organ cannot, and guard the by-construction trap (a spreading signal trivially reaches neighbours) by
making the HEADLINE a functional consequence at MATCHED total activity, with an uncoupling falsifier.

HONEST FINDING FIRST (a real result, reported not hidden): a FULLY REGENERATIVE Ca2+ wave (Ca2+-induced Ca2+
release) is all-or-nothing — once it ignites anywhere it FLOODS the whole array, so it does NOT discriminate
spatial patterns. The computation lives in the GRADED diffusive spread across the syncytium, not the regenerative
wave. We report all three regimes so the point is measured, not asserted.

  (A) SPATIAL-DENSITY GATE. At MATCHED total co-activity, the syncytium potentiates spatially-CLUSTERED co-active
      synapses (their Ca2+ pools above the plasticity threshold) but leaves the SAME NUMBER of SCATTERED ones
      sub-threshold — a density detector the point-wise organ has no access to.
  (B) HETEROSYNAPTIC BINDING. A silent-but-surrounded synapse (a gap inside a cluster) is potentiated by Ca2+
      pooled from its active neighbours — bound into the assembly (astrocyte-mediated heterosynaptic plasticity;
      Henneberger 2010; Andrade-Talavera). Point-wise leaves it silent.
  (C) FALSIFIERS: UNCOUPLE the syncytium (no spread) -> clustered ≈ scattered and no fill-in, at matched activity
      (so it is the coupling, not the activity). And the REGENERATIVE-WAVE control FLOODS (clustered ≈ scattered,
      both high) -> the selectivity is the graded spread, not an all-or-nothing wave.

    python -m src.eval.astrocyte_syncytium --seeds 5
"""
import argparse
import json
import math
import os

import torch

N = 120
K_ACTIVE = 18          # co-active synapses per trial (MATCHED between clustered and scattered)
N_GAP = 4              # silent-but-surrounded synapses inside a cluster
GATE_THR = 0.7
ACT = 0.15         # a SINGLE synapse's Ca2+ (ACT*TAU=0.6) is SUB-threshold -> a point cannot trigger plasticity;
TAU = 4.0          # only spatially-clustered co-activity POOLS enough Ca2+ across the syncytium to cross the gate


def astro_ca(activity, D, regen, steps=100, thr=0.6, dt=0.4, cap=3.0):
    """Astrocyte Ca2+ over a 1-D synapse array. Local activity gives a graded bump; it SPREADS by diffusion (D,
    the gap-junction syncytium) and, if regen>0, REGENERATES above threshold (a propagating wave). D=regen=0 =>
    point-wise (no spread)."""
    c = torch.zeros(N)
    for _ in range(steps):
        lap = torch.zeros(N); lap[1:-1] = c[:-2] + c[2:] - 2 * c[1:-1]
        regen_term = regen * torch.sigmoid(8 * (c - thr)) * (cap - c) / cap
        c = (c + (-c / TAU + D * lap + ACT * activity + regen_term) * dt).clamp(0, cap)
    return c


def gate(c):
    return torch.sigmoid(6 * (c - GATE_THR))


def clustered(gen):
    start = torch.randint(10, N - K_ACTIVE - 10, (1,), generator=gen).item()
    idx = torch.arange(start, start + K_ACTIVE + N_GAP)
    gaps = idx[torch.randperm(len(idx), generator=gen)[:N_GAP]]
    a = torch.zeros(N); a[idx] = 1.0; a[gaps] = 0.0
    return a, gaps


def scattered(gen):
    a = torch.zeros(N); a[torch.randperm(N, generator=gen)[:K_ACTIVE]] = 1.0
    return a


def regime_gates(D, regen, seed, trials=40):
    """Fraction of synapses POTENTIATED (plasticity gate > 0.5) at: clustered-active, scattered-active, and the
    gap (silent-but-surrounded) synapses. The fraction bound is cleaner than a mean gate (which cluster edges,
    pooling less, drag down)."""
    gc = gs = gf = 0.0
    for i in range(trials):
        g1 = torch.Generator().manual_seed(seed * 1000 + i)
        g2 = torch.Generator().manual_seed(seed * 1000 + i + 500000)
        ac, gaps = clustered(g1); asc = scattered(g2)
        cc = gate(astro_ca(ac, D, regen)); cs = gate(astro_ca(asc, D, regen))
        gc += (cc[ac > 0] > 0.5).float().mean().item()
        gs += (cs[asc > 0] > 0.5).float().mean().item()
        gf += (cc[gaps] > 0.5).float().mean().item()
    return gc / trials, gs / trials, gf / trials


def run_seed(seed):
    gc_u, gs_u, gf_u = regime_gates(0.0, 0.0, seed)        # UNCOUPLED (point-wise, no spread)
    gc_s, gs_s, gf_s = regime_gates(3.5, 0.0, seed)        # SYNCYTIUM (graded diffusive spread)
    gc_r, gs_r, gf_r = regime_gates(0.6, 1.5, seed)        # REGENERATIVE WAVE (floods)
    return {
        "gate_clustered_syncytium": round(gc_s, 4),
        "gate_scattered_syncytium": round(gs_s, 4),
        "gate_fillin_syncytium": round(gf_s, 4),
        "gate_clustered_uncoupled": round(gc_u, 4),
        "gate_scattered_uncoupled": round(gs_u, 4),
        "gate_fillin_uncoupled": round(gf_u, 4),
        "gate_clustered_regenwave": round(gc_r, 4),
        "gate_scattered_regenwave": round(gs_r, 4),
        "density_selectivity_syncytium": round(gc_s - gs_s, 4),      # clustered vs scattered (the density gate)
        "density_selectivity_uncoupled": round(gc_u - gs_u, 4),      # ~0 (falsifier: no spread, no selectivity)
        "density_selectivity_regenwave": round(gc_r - gs_r, 4),      # ~0 (floods -> no selectivity)
        "fillin_gap": round(gf_s - gf_u, 4),                         # heterosynaptic binding vs point-wise
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["gate_clustered_syncytium", "gate_scattered_syncytium", "gate_fillin_syncytium",
        "gate_clustered_uncoupled", "gate_scattered_uncoupled", "gate_fillin_uncoupled",
        "gate_clustered_regenwave", "gate_scattered_regenwave", "density_selectivity_syncytium",
        "density_selectivity_uncoupled", "density_selectivity_regenwave", "fillin_gap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: SYNCYTIUM clustered {p['gate_clustered_syncytium']:.2f} vs scattered "
              f"{p['gate_scattered_syncytium']:.2f} (fill-in {p['gate_fillin_syncytium']:.2f}) | uncoupled "
              f"{p['gate_clustered_uncoupled']:.2f}/{p['gate_scattered_uncoupled']:.2f} | regen-wave "
              f"{p['gate_clustered_regenwave']:.2f}/{p['gate_scattered_regenwave']:.2f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nASTROCYTE SYNCYTIUM — spatial-density-gated plasticity + heterosynaptic binding (n={a.seeds}; "
          f"mean ± 95% CI)\n" + "=" * 96, flush=True)
    print(f"  (A) SPATIAL-DENSITY GATE (matched total co-activity): CLUSTERED co-active synapses potentiate "
          f"{agg['gate_clustered_syncytium'][0]:.3f} vs SCATTERED {agg['gate_scattered_syncytium'][0]:.3f} "
          f"(selectivity {agg['density_selectivity_syncytium'][0]:+.3f} ± {agg['density_selectivity_syncytium'][1]:.3f})", flush=True)
    print(f"  (B) HETEROSYNAPTIC BINDING: a silent-but-surrounded gap synapse is bound in at gate "
          f"{agg['gate_fillin_syncytium'][0]:.3f} vs {agg['gate_fillin_uncoupled'][0]:.3f} point-wise (fill-in "
          f"{agg['fillin_gap'][0]:+.3f})", flush=True)
    print(f"  (C) FALSIFIERS: UNCOUPLED -> selectivity {agg['density_selectivity_uncoupled'][0]:+.3f} (no spread, "
          f"no density gate); REGENERATIVE WAVE FLOODS -> clustered {agg['gate_clustered_regenwave'][0]:.2f} ≈ "
          f"scattered {agg['gate_scattered_regenwave'][0]:.2f} (selectivity {agg['density_selectivity_regenwave'][0]:+.3f}) "
          f"— the computation is the GRADED spread, not the all-or-nothing wave.", flush=True)

    sound = (agg["fillin_gap"][0] > 0.5 and
             agg["density_selectivity_syncytium"][0] > 0.15 and
             abs(agg["density_selectivity_uncoupled"][0]) < 0.1 and
             agg["gate_clustered_uncoupled"][0] < 0.15 and agg["gate_scattered_regenwave"][0] > 0.6)
    verdict = ("SOUND — the astrocyte SYNCYTIUM binds a silent-but-surrounded synapse heterosynaptically into an "
               "assembly (fill-in the point-wise organ has no access to) and gates plasticity SELECTIVELY by "
               "spatial density (clustered co-activity's core potentiates, matched-count scattered does not), "
               "where an uncoupled astrocyte does nothing at this sub-threshold drive and a regenerative wave "
               "floods indiscriminately. A network computation from glial coupling, against its falsifiers — the "
               "effect is real but modest (only the cluster core binds; the graded spread, not the wave, does the "
               "work)." if sound
               else "WEAK — the binding / density-gate did not clear the uncoupled and flood controls; revisit.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "k_active": K_ACTIVE, "gate_thr": GATE_THR,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/astrocyte_syncytium.json", "w"), indent=2)
    _svg(agg, "results/astrocyte_syncytium.svg")
    print("\nwrote results/astrocyte_syncytium.json and results/astrocyte_syncytium.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Astrocyte syncytium: plasticity gated by spatial density of co-activity</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">clustered co-active synapses pool enough Ca&#178; '
             'to potentiate where the same number scattered do not; a point-wise astrocyte can’t tell them apart, '
             'a regenerative wave floods</text>')
    oy = 60; base = oy + ph
    # Panel A: clustered vs scattered gate across three regimes
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) potentiation gate: clustered vs scattered</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    trip = [("syncytium", agg["gate_clustered_syncytium"][0], agg["gate_scattered_syncytium"][0]),
            ("uncoupled", agg["gate_clustered_uncoupled"][0], agg["gate_scattered_uncoupled"][0]),
            ("regen wave", agg["gate_clustered_regenwave"][0], agg["gate_scattered_regenwave"][0])]
    for i, (lab, cl, sc) in enumerate(trip):
        x = oxA + 18 + i * 78
        for j, (v, col) in enumerate([(cl, "#2ca25f"), (sc, "#c9341a")]):
            hh = v * (ph - 24); xx = x + j * 26
            e.append(f'<rect x="{xx}" y="{base-hh:.1f}" width="22" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
            e.append(f'<text x="{xx+11}" y="{base-hh-4:.0f}" font-size="8.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        e.append(f'<text x="{x+24}" y="{base+13:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<text x="{oxA}" y="{base+30:.0f}" font-size="9" fill="#2ca25f">green = clustered</text>')
    e.append(f'<text x="{oxA+110}" y="{base+30:.0f}" font-size="9" fill="#c9341a">red = scattered (matched count)</text>')
    e.append(f'<text x="{oxA}" y="{base+42:.0f}" font-size="9" fill="#5b6b8c">only the syncytium separates them; uncoupled=both low, regen wave=both flood</text>')
    # Panel B: heterosynaptic fill-in (syncytium vs uncoupled)
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) heterosynaptic binding of a silent gap</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    b2 = [("syncytium\n(bound)", agg["gate_fillin_syncytium"][0], "#2ca25f"), ("point-wise\n(silent)", agg["gate_fillin_uncoupled"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(b2):
        hh = v * (ph - 24); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-hh:.1f}" width="64" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+32}" y="{base-hh-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">a silent synapse surrounded by active ones is recruited by pooled neighbour Ca&#178;</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
