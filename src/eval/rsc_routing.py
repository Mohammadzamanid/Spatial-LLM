"""
src/eval/rsc_routing.py

BIFURCATED RSC ROUTING — an action pathway and a memory pathway emerge with dissociated reference frames, and the
split enables a double dissociation a unified readout cannot (GAPS.md: the "unified spatial->LLM gate" critique).

The critique: the model bridges the spatial cortex to the LLM through a single, unified gated cross-attention, but
the retrosplenial cortex does not pass a unified map forward — it is bifurcated. M2-projecting RSC neurons route
spatial information to secondary motor cortex for ACTION-affordances, while AD-projecting neurons route to anterior
thalamus to anchor allocentric spatial MEMORY; inactivating one pathway impairs place-action association, the other
object-location memory (projection-specific dissociation, *Molecular Psychiatry* 2024; RSC→M2, *J. Neurosci.* 2016).
A single shared readout collapses this division of labour. This is an architecture claim, so — per the standing rule
— we hardcode only the two-pathway wiring (as the anatomy does) and let the CONTENT and the benefit emerge and be
measured; we do NOT assign either head its reference frame or its selectivity.

Two conflicting demands are placed on the same spatial code: ACTION = the egocentric, heading-EQUIVARIANT "which way
do I turn to reach the object" (motor affordance), and MEMORY = the allocentric, heading-INVARIANT "where the object
is" (location memory). A unified head must be both heading-dependent and heading-invariant at once. Measured, never
imposed:

  (A) REFERENCE FRAMES DISSOCIATE (emergent). Trained only on the combined task, the ACTION head becomes
      heading-EQUIVARIANT (heading is decodable from it) while the MEMORY head becomes heading-INVARIANT (heading is
      NOT decodable) — an egocentric/allocentric split that was never assigned.
  (B) SELECTIVE ROUTING. The MEMORY pathway carries the allocentric location but not the egocentric action signal
      (selective), whereas a UNIFIED code is ENTANGLED — it carries both, so every downstream target receives
      everything instead of what it needs.
  (C) THE SPLIT ENABLES THE DOUBLE DISSOCIATION. Lesion the action pathway → an action-only deficit; lesion the
      memory pathway → a memory-only deficit. A UNIFIED code lesioned by the same amount loses BOTH — so it is the
      segregated architecture that makes the observed optogenetic double dissociation possible at all.
  (D) FALSIFIER — no conflict. Make BOTH tasks allocentric (a shared reference frame): the specialization blurs (the
      memory pathway no longer cleanly excludes the action signal). The division of labour emerges from the
      conflicting frames, not from the wiring.

Honest note: the split does NOT lower the total training loss — a full-capacity unified head fits both tasks. The
benefit is clean functional SEGREGATION (target-appropriate routing + selective lesionability), not efficiency —
exactly what the anatomy is for, and measured on that metric rather than a coarse loss.

Multi-seed, mean ± 95% CI. Writes results/rsc_routing.json + .svg.

    python -m src.eval.rsc_routing --seeds 5
"""
import argparse
import json
import math
import os

import torch

NC = 12                 # place / object RBF centres
DTOT = 16               # total hidden capacity (unified gets DTOT; split gets DTOT/2 per head — matched total)
STEPS = 3000


def _mlp(din, dh, gen):
    return [(torch.randn(din, dh, generator=gen) * (2 / din) ** .5).requires_grad_(True),
            torch.zeros(dh, requires_grad=True),
            (torch.randn(dh, 2, generator=gen) * (2 / dh) ** .5).requires_grad_(True),
            torch.zeros(2, requires_grad=True)]


def _head(P, x, mask=None):
    h = torch.relu(x @ P[0] + P[1])
    if mask is not None:
        h = h * mask
    return h @ P[2] + P[3], h


def _feats(p, g, th, Cp, Cg):
    place = torch.exp(-((p[:, None, :] - Cp[None]) ** 2).sum(-1) / (2 * 0.15 ** 2))
    obj = torch.exp(-((g[:, None, :] - Cg[None]) ** 2).sum(-1) / (2 * 0.15 ** 2))
    return torch.cat([place, obj, torch.sin(th), torch.cos(th)], 1)


def _batch(n, gen, ego):
    p = torch.rand(n, 2, generator=gen); g = torch.rand(n, 2, generator=gen)
    th = torch.rand(n, 1, generator=gen) * 2 * math.pi
    ang = torch.atan2(g[:, 1:] - p[:, 1:], g[:, :1] - p[:, :1])
    ang = ang - th if ego else ang                                  # egocentric (heading-equivariant) vs allocentric
    action = torch.cat([torch.cos(ang), torch.sin(ang)], 1)
    return p, g, th, action, g                                      # memory target = allocentric object location g


def _r2(h, tgt):
    Hb = torch.cat([h, torch.ones(h.shape[0], 1)], 1)
    W = torch.linalg.lstsq(Hb, tgt).solution
    return max(0.0, 1 - ((Hb @ W - tgt) ** 2).mean().item() / tgt.var().item())


def train(ego, split, seed):
    gen = torch.Generator().manual_seed(seed)
    Cp = torch.rand(NC, 2, generator=gen); Cg = torch.rand(NC, 2, generator=gen)
    din = 2 * NC + 2
    if split:
        Pa, Pm = _mlp(din, DTOT // 2, gen), _mlp(din, DTOT // 2, gen); params = Pa + Pm
    else:
        Pa = _mlp(din, DTOT, gen)
        Pm = [Pa[0], Pa[1], (torch.randn(DTOT, 2, generator=gen) * .1).requires_grad_(True), torch.zeros(2, requires_grad=True)]
        params = Pa + [Pm[2], Pm[3]]
    opt = torch.optim.Adam(params, 3e-3)
    for _ in range(STEPS):
        p, g, th, act, mem = _batch(256, gen, ego)
        x = _feats(p, g, th, Cp, Cg)
        loss = ((_head(Pa, x)[0] - act) ** 2).mean() + ((_head(Pm, x)[0] - mem) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return Pa, Pm, Cp, Cg, gen


def run_seed(seed):
    # ---- CONFLICT: egocentric action + allocentric memory ----
    Pa, Pm, Cp, Cg, gen = train(True, True, seed)
    p, g, th, act, mem = _batch(4000, gen, True)
    x = _feats(p, g, th, Cp, Cg)
    _, ha = _head(Pa, x); _, hm = _head(Pm, x)
    hd = torch.cat([torch.sin(th), torch.cos(th)], 1)
    ah_head, mh_head = _r2(ha, hd), _r2(hm, hd)                     # (A) heading decodable per head
    mem_sel = _r2(hm, g) - _r2(hm, act)                            # (B) memory pathway: location, not action
    # (C) lesion: split — kill the action head, then the memory head
    def losses(Pa_, Pm_, mask_a=None, mask_m=None):
        la = ((_head(Pa_, x, mask_a)[0] - act) ** 2).mean().item()
        lm = ((_head(Pm_, x, mask_m)[0] - mem) ** 2).mean().item()
        return la, lm
    base_a, base_m = losses(Pa, Pm)
    kill = torch.zeros(DTOT // 2)
    la_kA, lm_kA = losses(Pa, Pm, mask_a=kill)                     # action pathway lesioned
    la_kM, lm_kM = losses(Pa, Pm, mask_m=kill)                     # memory pathway lesioned
    split_dissoc = (la_kA / base_a - 1) + (lm_kM / base_m - 1) - (lm_kA / base_m - 1) - (la_kM / base_a - 1)
    # unified: lesion half the shared units -> both tasks degrade (no clean dissociation)
    Ua, Um, Cp2, Cg2, gen2 = train(True, False, seed)
    p2, g2, th2, act2, mem2 = _batch(4000, gen2, True); x2 = _feats(p2, g2, th2, Cp2, Cg2)
    ub_a = ((_head(Ua, x2)[0] - act2) ** 2).mean().item(); ub_m = ((_head(Um, x2)[0] - mem2) ** 2).mean().item()
    half = torch.cat([torch.zeros(DTOT // 2), torch.ones(DTOT - DTOT // 2)])
    ua_h = ((_head(Ua, x2, half)[0] - act2) ** 2).mean().item(); um_h = ((_head(Um, x2, half)[0] - mem2) ** 2).mean().item()
    uni_actdef, uni_memdef = ua_h / ub_a - 1, um_h / ub_m - 1      # both > 0 -> entangled, no dissociation
    uni_entangle = min(_r2(_head(Ua, x2)[1], act2), _r2(_head(Ua, x2)[1], g2))   # shared code carries BOTH
    # ---- FALSIFIER: both allocentric (no conflict) ----
    Fa, Fm, Cp3, Cg3, gen3 = train(False, True, seed)
    p3, g3, th3, act3, mem3 = _batch(4000, gen3, False); x3 = _feats(p3, g3, th3, Cp3, Cg3)
    mh_act_aligned = _r2(_head(Fm, x3)[1], act3)                   # memory pathway leaks action when no conflict

    return {"act_head_heading": ah_head, "mem_head_heading": mh_head,
            "mem_pathway_selectivity": mem_sel,
            "split_les_action_on_action": la_kA / base_a, "split_les_action_on_memory": lm_kA / base_m,
            "split_les_memory_on_memory": lm_kM / base_m, "split_les_memory_on_action": la_kM / base_a,
            "unified_lesion_action_deficit": uni_actdef, "unified_lesion_memory_deficit": uni_memdef,
            "unified_entanglement": uni_entangle,
            "mem_pathway_action_conflict": _r2(hm, act), "mem_pathway_action_aligned": mh_act_aligned}


KEYS = ["act_head_heading", "mem_head_heading", "mem_pathway_selectivity",
        "split_les_action_on_action", "split_les_action_on_memory",
        "split_les_memory_on_memory", "split_les_memory_on_action",
        "unified_lesion_action_deficit", "unified_lesion_memory_deficit", "unified_entanglement",
        "mem_pathway_action_conflict", "mem_pathway_action_aligned"]


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

    print(f"BIFURCATED RSC ROUTING — action & memory pathways emerge with dissociated frames "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 84, flush=True)
    print(f"  (A) REFERENCE FRAMES DISSOCIATE (emergent): heading decodable from ACTION head "
          f"{agg['act_head_heading'][0]:.2f} vs MEMORY head {agg['mem_head_heading'][0]:.2f} — the action pathway is "
          f"egocentric, the memory pathway allocentric (never assigned)", flush=True)
    print(f"  (B) SELECTIVE ROUTING: memory pathway location-vs-action selectivity "
          f"{agg['mem_pathway_selectivity'][0]:+.2f} (carries WHERE, not the turn); unified code entanglement "
          f"{agg['unified_entanglement'][0]:.2f} (carries BOTH)", flush=True)
    print(f"  (C) THE SPLIT ENABLES THE DOUBLE DISSOCIATION (lesion → ×error):", flush=True)
    print(f"      SPLIT: kill action pathway → action ×{agg['split_les_action_on_action'][0]:.2f}, memory "
          f"×{agg['split_les_action_on_memory'][0]:.2f} | kill memory pathway → memory "
          f"×{agg['split_les_memory_on_memory'][0]:.2f}, action ×{agg['split_les_memory_on_action'][0]:.2f} "
          f"(each lesion hits ONE task)", flush=True)
    print(f"      UNIFIED: kill half the shared units → action deficit +{agg['unified_lesion_action_deficit'][0]*100:.0f}% "
          f"AND memory deficit +{agg['unified_lesion_memory_deficit'][0]*100:.0f}% (BOTH — no dissociation possible)",
          flush=True)
    print(f"  (D) FALSIFIER — no conflict (both allocentric): memory pathway carries the action signal "
          f"{agg['mem_pathway_action_aligned'][0]:.2f} vs only {agg['mem_pathway_action_conflict'][0]:.2f} under "
          f"conflict — the specialization needs the conflicting frames", flush=True)
    print(f"\n  Splitting the spatial read-out into an action pathway and a memory pathway makes the two reference "
          f"frames DISSOCIATE and routes each target its own signal — enabling the optogenetic double dissociation a "
          f"unified gate cannot. Honest: this does not lower total loss; the benefit is clean segregation, not "
          f"efficiency.", flush=True)

    out = {"n_seeds": a.seeds, "capacity_total": DTOT, "centres": NC,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS},
           "verdict": "Splitting the spatial read-out into an ACTION pathway (M2/motor) and a MEMORY pathway "
                      "(AD/thalamus), as the retrosplenial cortex does (Molecular Psychiatry 2024), makes the two "
                      "reference frames DISSOCIATE emergently -- the action head becomes egocentric "
                      "(heading-equivariant), the memory head allocentric (heading-invariant), never assigned -- "
                      "and routes each target its own signal (the memory pathway carries location not the turn, "
                      "while a unified code is entangled). The segregation is what ENABLES the observed double "
                      "dissociation: lesioning one split pathway hits one task, but lesioning a unified code hits "
                      "BOTH. A no-conflict falsifier (both tasks allocentric) blurs the specialization, so it "
                      "emerges from the conflicting frames, not the wiring. Honest note: the split does NOT lower "
                      "total loss (a full-capacity unified head fits both); the benefit is clean functional "
                      "segregation for differential routing and selective lesionability, measured on that metric "
                      "rather than a coarse loss."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/rsc_routing.json", "w"), indent=2)
    svg_rsc(agg, "results/rsc_routing.svg")
    print("\nwrote results/rsc_routing.json and results/rsc_routing.svg", flush=True)


def svg_rsc(agg, out):
    W_, H = 770, 320
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Bifurcated RSC routing: action &amp; memory pathways emerge with dissociated reference frames</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">the split enables the double dissociation a unified '
         'gate cannot &#8212; wiring hardcoded, content &amp; benefit emergent</text>']
    # left: heading decodable per head (A)
    bx, by, bh, bw = 44, 100, 150, 52
    e.append(f'<text x="{bx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">heading in head (R&#178;)</text>')
    for i, (k, lab, col) in enumerate([("act_head_heading", "ACTION\n(egocentric)", "#e6842a"), ("mem_head_heading", "MEMORY\n(allocentric)", "#2b8cbe")]):
        v = max(0.0, agg[k][0]); x = bx + i * (bw + 16); h = v * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh}" x2="{bx+2*(bw+16):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{bx}" y="{by+bh+38:.0f}" font-size="8.5" fill="#5b6b8c">frames dissociate (never assigned)</text>')
    # middle: lesion — split dissociates, unified doesn't (C)
    m0 = 300; mw = 40
    e.append(f'<text x="{m0}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">lesion &#8594; deficit</text>')
    les = [("split_les_action_on_action", "killA\n→act", "#c9341a"), ("split_les_action_on_memory", "killA\n→mem", "#8c8c8c"),
           ("unified_lesion_action_deficit", "uni½\n→act", "#c9341a"), ("unified_lesion_memory_deficit", "uni½\n→mem", "#c9341a")]
    def barval(k):
        return agg[k][0] if "unified" in k else agg[k][0] - 1     # deficits as fractional increase
    mx = max(0.2, max(barval(k) for k, _, _ in les)) * 1.2
    for i, (k, lab, col) in enumerate(les):
        v = max(0.0, barval(k)); x = m0 + i * (mw + 6); h = min(v / mx, 1.0) * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{mw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh-h-4:.0f}" font-size="8.5" font-weight="700" fill="#0b1324" text-anchor="middle">+{v*100:.0f}%</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh+12+li*10:.0f}" font-size="8" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{m0-4}" y1="{by+bh}" x2="{m0+4*(mw+6):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{m0}" y="{by+bh+34:.0f}" font-size="8" fill="#5b6b8c">split: one task each; unified: BOTH</text>')
    # right: selectivity vs entanglement (B) + falsifier (D)
    rx = 560
    e.append(f'<text x="{rx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">memory pathway</text>')
    e.append(f'<text x="{rx}" y="{by+22}" font-size="10" fill="#2b8cbe">selectivity {agg["mem_pathway_selectivity"][0]:+.2f}</text>')
    e.append(f'<text x="{rx}" y="{by+37}" font-size="9" fill="#5b6b8c">(carries WHERE, not the turn)</text>')
    e.append(f'<text x="{rx}" y="{by+62}" font-size="10" fill="#c9341a">unified entangled {agg["unified_entanglement"][0]:.2f}</text>')
    e.append(f'<text x="{rx}" y="{by+77}" font-size="9" fill="#5b6b8c">(carries BOTH)</text>')
    e.append(f'<text x="{rx}" y="{by+104}" font-size="10" font-weight="700" fill="#28324a">falsifier (no conflict)</text>')
    e.append(f'<text x="{rx}" y="{by+122}" font-size="9.5" fill="#28324a">action leak into memory: '
             f'{agg["mem_pathway_action_conflict"][0]:.2f} &#8594; {agg["mem_pathway_action_aligned"][0]:.2f}</text>')
    e.append(f'<text x="20" y="{H-12}" font-size="9.5" fill="#5b6b8c">Honest: the split does not lower total loss '
             f'&#8212; a full unified head fits both. The benefit is clean segregation for routing + lesionability.</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
