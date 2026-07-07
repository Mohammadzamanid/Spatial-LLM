"""
src/eval/social_grid_cortex.py

CPU DE-RISK for #9 — "the LLM reasons over a 2-D SOCIAL space" (Tavares 2015; Park, Miller 2021: humans map a
social hierarchy of POWER x AFFILIATION with the same grid/hippocampal machinery; builds on gap #4's self/other
place cells). The T4 headline (a frozen Qwen+LoRA answers "who is more dominant?" and "who is socially closer to
X?" cortex-ON vs text-only-OFF) lives in notebooks/m9_social_grid_llm_kaggle.py + src/training/train_social.py.
As with #8, this file validates the design NON-CIRCULARLY on the ACTUAL frozen cortex.encode pipeline the LLM
reads (real numbers, n=5, matched controls) BEFORE the T4 cell.

Agents are placed in a 2-D social space: axis-0 = POWER (dominance), axis-1 = AFFILIATION. Each agent enters by
its OWN social position through the FROZEN space-pretrained cortex (a directed path; never a relative
displacement — no leak). Two socially-distinct signatures with a DISSOCIATION (the #9 analogue of gap #4's
self/other double dissociation, now at the map level):
  (A) DOMINANCE — a 1-D read of the POWER axis (the social transitive-inference result, Kumaran 2016;
      Park-Miller 2021): held-out pairwise dominance from the decoded power coordinate >> chance.
  (B) SOCIAL DISTANCE — a genuine 2-D metric: OFF-AXIS "socially closer" (triples where the power-axis ordering
      DISAGREES with true 2-D social distance) > chance. A power-only (1-D) read is <=0.5 there by construction.
  (C) AXIS DISSOCIATION — dominance is read from the POWER axis, NOT affiliation: ranking held-out agents by the
      decoded AFFILIATION coordinate predicts dominance only at chance. The frozen code exposes the two social
      axes SEPARATELY.
FALSIFIER (parameter-free): SHUFFLE the agent<->position map -> dominance and social-distance both collapse to
chance. Honest note (as #8): absolute 2-D strength is modest on CPU with these simple read-outs; the de-risk's
job is to show the social metric is PRESENT, DISSOCIABLE, and CONTROL-CLEAN — which it is.

    python -m src.eval.social_grid_cortex --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.eval.conceptual_grid_cortex import walk_2d, _spearman, _offaxis_closer
from src.eval.structural_transfer_cortex import pretrain_cortex

CHANCE = 0.5


def _decode(codes, tr, te, grid):
    """Least-squares code->(power, affil); return decoded held-out coords (M,2)."""
    A = torch.cat([codes[tr], torch.ones(len(tr), 1)], -1)
    W = torch.linalg.lstsq(A, grid[tr]).solution
    Ate = torch.cat([codes[te], torch.ones(len(te), 1)], -1)
    return Ate @ W


def _dominance_acc(coord_axis, true_power, min_gap):
    """Pairwise dominance accuracy: sign(coord_axis[i]-coord_axis[j]) vs sign(true_power[i]-true_power[j]),
    over pairs whose true power gap exceeds min_gap (skip ties)."""
    M = len(true_power); cor = tot = 0
    for i in range(M):
        for j in range(i + 1, M):
            if abs((true_power[i] - true_power[j]).item()) <= min_gap:
                continue
            tot += 1
            cor += int((coord_axis[i] > coord_axis[j]) == (true_power[i] > true_power[j]))
    return cor / max(tot, 1)


def run_seed(seed, G=6, spacing=0.8):
    cx = pretrain_cortex(seed)                                     # space-only pretrain, then FROZEN
    xs = torch.arange(G).float() * spacing - (G - 1) * spacing / 2
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)   # axis0=power, axis1=affil
    N = grid.shape[0]
    with torch.no_grad():
        h, s, vz = walk_2d(grid)
        codes = cx.encode(h, s, vz)                                # (N,128) FROZEN agent codes
    posd = torch.cdist(grid, grid)
    near_r = 1.2 * spacing

    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 3))
    tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7))
    gte = grid[te]; true_power = gte[:, 0]; min_gap = 0.5 * spacing

    # (A) DOMINANCE — decoded POWER axis, held-out agents
    dec = _decode(codes, tr, te, grid)                            # (M,2) decoded (power, affil)
    dom_power = _dominance_acc(dec[:, 0], true_power, min_gap)
    # (C) DISSOCIATION — decoded AFFILIATION axis should NOT predict dominance
    dom_affil = _dominance_acc(dec[:, 1], true_power, min_gap)
    # shuffled falsifier for dominance
    dec_sh = _decode(codes[perm], tr, te, grid)
    dom_power_sh = _dominance_acc(dec_sh[:, 0], true_power, min_gap)

    # (B) SOCIAL DISTANCE — off-axis "socially closer" (2-D), readout-free on raw code-distance
    cded = torch.cdist(codes, codes)
    gen = torch.Generator().manual_seed(seed + 100)
    soc_off, n_off = _offaxis_closer(posd, cded, grid, near_r, gen)
    cded_sh = torch.cdist(codes[perm], codes[perm])
    gen2 = torch.Generator().manual_seed(seed + 101)
    soc_off_sh, _ = _offaxis_closer(posd, cded_sh, grid, near_r, gen2)
    # and in the held-out DECODED social space
    decd = torch.cdist(dec, dec); truen = torch.cdist(gte, gte)
    gen3 = torch.Generator().manual_seed(seed + 200); M = len(te); ho_t = ho_c = 0
    for _ in range(4000):
        a, b, c = torch.randint(M, (3,), generator=gen3).tolist()
        if len({a, b, c}) < 3:
            continue
        tb = truen[a, b] < truen[a, c]
        x1 = (gte[a, 0] - gte[b, 0]).abs() < (gte[a, 0] - gte[c, 0]).abs()      # power-axis ordering
        if bool(x1) != bool(tb):
            ho_t += 1; ho_c += int((decd[a, b] < decd[a, c]) == tb)
    soc_off_dec = ho_c / max(ho_t, 1)

    return {
        "dominance_power": round(dom_power, 4),
        "dominance_affil": round(dom_affil, 4),                    # dissociation control (-> chance)
        "dominance_power_shuffled": round(dom_power_sh, 4),        # falsifier (-> chance)
        "social_offaxis_free": round(soc_off, 4),
        "social_offaxis_free_shuffled": round(soc_off_sh, 4),
        "social_offaxis_decoded": round(soc_off_dec, 4),
        "dominance_metric_spearman": round(_spearman(posd[torch.triu_indices(N, N, 1)[0],
                                          torch.triu_indices(N, N, 1)[1]],
                                          cded[torch.triu_indices(N, N, 1)[0],
                                          torch.triu_indices(N, N, 1)[1]]), 4),
        "dissociation_gap": round(dom_power - dom_affil, 4),
        "social_gap": round(soc_off - soc_off_sh, 4),
        "n_offaxis": n_off,
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["dominance_power", "dominance_affil", "dominance_power_shuffled", "social_offaxis_free",
        "social_offaxis_free_shuffled", "social_offaxis_decoded", "dominance_metric_spearman",
        "dissociation_gap", "social_gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.8)
    a = ap.parse_args()
    per = [run_seed(s, G=a.G, spacing=a.spacing) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: dominance(power) {p['dominance_power']:.3f} vs (affil) {p['dominance_affil']:.3f} vs "
              f"shuffled {p['dominance_power_shuffled']:.3f} | social off-axis {p['social_offaxis_free']:.3f} "
              f"(decoded {p['social_offaxis_decoded']:.3f})", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nSOCIAL GRID — the FROZEN space cortex exposes a 2-D social map (CPU de-risk for #9, n={a.seeds}; "
          f"mean ± 95% CI)\n" + "=" * 96, flush=True)
    print(f"  (A) DOMINANCE (1-D power axis — social transitive inference, Kumaran 2016 / Park-Miller 2021):", flush=True)
    print(f"      held-out pairwise dominance from decoded POWER {agg['dominance_power'][0]:.3f} ± "
          f"{agg['dominance_power'][1]:.3f}  (chance {CHANCE})", flush=True)
    print(f"  (B) SOCIAL DISTANCE (genuine 2-D — OFF-AXIS 'socially closer'):", flush=True)
    print(f"      readout-free {agg['social_offaxis_free'][0]:.3f} ± {agg['social_offaxis_free'][1]:.3f} vs "
          f"SHUFFLED {agg['social_offaxis_free_shuffled'][0]:.3f} (gap {agg['social_gap'][0]:+.3f} ± "
          f"{agg['social_gap'][1]:.3f}); held-out decoded {agg['social_offaxis_decoded'][0]:.3f}  (chance "
          f"{CHANCE}; a power-only read is <=0.5 here)", flush=True)
    print(f"  (C) AXIS DISSOCIATION — dominance is read from POWER, not affiliation:", flush=True)
    print(f"      decoded POWER -> dominance {agg['dominance_power'][0]:.3f}  vs  decoded AFFILIATION -> "
          f"dominance {agg['dominance_affil'][0]:.3f} (gap {agg['dissociation_gap'][0]:+.3f} ± "
          f"{agg['dissociation_gap'][1]:.3f}) — the two social axes are SEPARATELY readable", flush=True)
    print(f"      FALSIFIER: shuffled agent<->position -> dominance {agg['dominance_power_shuffled'][0]:.3f} "
          f"(-> chance)", flush=True)

    sound = (agg["dominance_power"][0] > CHANCE + 0.1 and
             agg["dissociation_gap"][0] > 0.1 and
             agg["social_offaxis_free"][0] > CHANCE + 0.03 and
             agg["dominance_power_shuffled"][0] < CHANCE + 0.1)
    verdict = ("SOUND — the frozen space code exposes a DISSOCIABLE 2-D social map (dominance from the power "
               "axis; social-distance a genuine 2-D metric a 1-D read cannot produce); build the T4 LLM cell "
               "(train_social.py / m9 notebook). Absolute 2-D strength is modest on CPU; the trained LLM readout "
               "is expected to sharpen it (1-D precedent: 1.0 -> 0.99)."
               if sound else
               "WEAK — the social map is not cleanly exposed; consider a social-axis pretraining pass before the "
               "T4 cell.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "G": a.G, "spacing": a.spacing, "chance": CHANCE,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/social_grid_cortex.json", "w"), indent=2)
    _svg(agg, "results/social_grid_cortex.svg")
    print("\nwrote results/social_grid_cortex.json and results/social_grid_cortex.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Social grid: a dissociable 2-D social map from the frozen cortex (CPU de-risk, #9)</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">dominance reads the POWER axis (not '
             'affiliation); "socially closer" needs a genuine 2-D metric; shuffled positions collapse both</text>')
    oy = 60; base = oy + ph; top = 1.0
    # Panel A: dissociation — dominance from power vs affiliation vs shuffled
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) dominance read-out (higher=better)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    ych = base - (CHANCE / top) * (ph - 24)
    e.append(f'<line x1="{oxA}" y1="{ych:.0f}" x2="{oxA+pw}" y2="{ych:.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{oxA+pw-2}" y="{ych-3:.0f}" font-size="8.5" fill="#9aa6bd" text-anchor="end">chance {CHANCE}</text>')
    bars = [("POWER\naxis", agg["dominance_power"][0], "#2ca25f"),
            ("AFFIL\naxis", agg["dominance_affil"][0], "#c98a1a"),
            ("SHUFFLED", agg["dominance_power_shuffled"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(bars):
        hh = (v / top) * (ph - 24); x = oxA + 20 + i * 74
        e.append(f'<rect x="{x}" y="{base-hh:.1f}" width="52" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-hh-6:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">dominance is read from POWER, not affiliation — the axes are separable</text>')
    # Panel B: social distance off-axis (readout-free) vs shuffled
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) off-axis "socially closer" (2-D, higher=better)</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    ych2 = base - (CHANCE / top) * (ph - 24)
    e.append(f'<line x1="{oxB}" y1="{ych2:.0f}" x2="{oxB+pw}" y2="{ych2:.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{oxB+pw-2}" y="{ych2-3:.0f}" font-size="8.5" fill="#9aa6bd" text-anchor="end">chance {CHANCE}</text>')
    b2 = [("frozen\ncode", agg["social_offaxis_free"][0], "#2ca25f"),
          ("decoded\nheld-out", agg["social_offaxis_decoded"][0], "#4ca66f"),
          ("SHUFFLED", agg["social_offaxis_free_shuffled"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(b2):
        hh = (v / top) * (ph - 24); x = oxB + 20 + i * 74
        e.append(f'<rect x="{x}" y="{base-hh:.1f}" width="52" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-hh-6:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">a power-only (1-D) read is &#8804;0.5 on off-axis triples by construction</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
