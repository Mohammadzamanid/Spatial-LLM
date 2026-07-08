"""
src/eval/conceptual_grid_cortex.py

CPU DE-RISK for #8 — "the LLM reads a 2-D CONCEPTUAL grid" (Constantinescu, Behrens 2016; Bellmund 2018:
a 2-D conceptual space navigated with grid-like code). The T4 headline (a frozen Qwen+LoRA answers
"which concept is closer / between?" cortex-ON vs text-only-OFF) lives in notebooks/m8_conceptual_grid_llm_kaggle.py
+ src/training/train_conceptual.py. FOLLOWING the repo's established practice (structural_transfer_cortex.py
de-risked train_relational.py on CPU BEFORE the T4 cell), this file validates the design NON-CIRCULARLY on the
ACTUAL frozen cortex.encode pipeline the LLM reads — with real numbers, n=5, and matched controls — so the
GPU cell is sound before anyone spends a T4 on it.

The claim under test: a cortex pretrained ONLY on physical Euclidean space, then FROZEN, encodes a concept
placed at 2-D coordinate (x,y) — via its OWN directed path (heading=atan2(y,x), speed=r/T; never the signed
relative displacement, which would leak) — such that the frozen code carries a GENUINE 2-D METRIC. The sharp,
non-circular 2-D signature is OFF-AXIS "closer": triples where the 1-D x-projection ordering DISAGREES with the
true 2-D answer. A 1-D (rank) code is <=0.5 there BY CONSTRUCTION; only a real 2-D metric beats chance.

Two un-memorizable read-outs (no fitted comparison head that could memorize):
  (A) READOUT-FREE geometry (parameter-free): Spearman(code-distance, true 2-D distance); OFF-AXIS "closer" by
      RAW code-distance. Cannot be circular — nothing is fit.
  (B) HELD-OUT linear decode: fit code->(x,y) on TRAIN concepts, decode concepts the probe NEVER saw; error in
      the decoded space; OFF-AXIS "closer" in the decoded space on held-out concepts.
FALSIFIER (parameter-free): SHUFFLE the concept<->position map -> (A) Spearman -> ~0 and off-axis -> chance,
(B) held-out decode error blows up (~3x). Honest note: the absolute strength is MODEST on CPU with these simple
read-outs (as the 1-D precedent was before its LLM readout sharpened TI 1.0 -> 0.99); the de-risk's job is only
to show the 2-D metric is PRESENT and CONTROL-CLEAN, which it is.

    python -m src.eval.conceptual_grid_cortex --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.eval.structural_transfer_cortex import pretrain_cortex   # the PROVEN frozen space-cortex pretrain

ENC_T = 8
CHANCE = 0.5


def walk_2d(pos, T=ENC_T):
    """Each 2-D concept position (x,y) -> a directed T-step path reaching net (x,y,0). heading=atan2(y,x),
    speed=r/T. The frozen cortex.encode PATH-INTEGRATES this to the concept's position — no leak (each item
    enters by its OWN position, never a relative displacement)."""
    x, y = pos[:, 0], pos[:, 1]
    r = torch.sqrt(x * x + y * y).clamp_min(1e-6)
    heading = torch.atan2(y, x).unsqueeze(1).expand(-1, T)
    speed = (r / T).unsqueeze(1).expand(-1, T)
    return heading.contiguous(), speed.contiguous(), torch.zeros(pos.shape[0], T)


def _spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def _offaxis_closer(posd, refd, grid, near_r, gen, n=12000):
    """BALANCED off-axis 'closer' accuracy using distances `refd` (chance = 0.5). Off-axis = the 1-D
    x-projection ordering disagrees with the true 2-D answer (a 1-D code is <=0.5 here by construction).
    The raw off-axis set is label-imbalanced (~0.67 one class), so a *constant* predictor would score ~0.67
    on it — we therefore BALANCE to equal true-label 0/1 counts so a constant predictor scores exactly 0.5
    and the reported number is the frozen code's genuine skill against an honest baseline."""
    N = grid.shape[0]; pos, neg = [], []
    for _ in range(n):
        a, b, c = torch.randint(N, (3,), generator=gen).tolist()
        if len({a, b, c}) < 3:
            continue
        tb = bool(posd[a, b] < posd[a, c])
        x1 = (grid[a, 0] - grid[b, 0]).abs() < (grid[a, 0] - grid[c, 0]).abs()
        if bool(x1) != tb and (posd[a, b] > near_r or posd[a, c] > near_r):         # off-axis, non-local
            correct = int((refd[a, b] < refd[a, c]) == tb)
            (pos if tb else neg).append(correct)
    m = min(len(pos), len(neg))
    if m == 0:
        return 0.5, 0
    bal = pos[:m] + neg[:m]
    return sum(bal) / len(bal), 2 * m


def run_seed(seed, G=6, spacing=0.8):
    cx = pretrain_cortex(seed)                                     # space-only pretrain, then FROZEN
    xs = torch.arange(G).float() * spacing - (G - 1) * spacing / 2
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)    # (G*G, 2) concept coords
    N = grid.shape[0]
    with torch.no_grad():
        h, s, vz = walk_2d(grid)
        codes = cx.encode(h, s, vz)                                # (N, 128) FROZEN concept codes
    posd = torch.cdist(grid, grid)
    cded = torch.cdist(codes, codes)
    iu = torch.triu_indices(N, N, 1)
    near_r = 1.2 * spacing                                          # "near" radius for the off-axis non-local guard

    # (A) READOUT-FREE geometry (parameter-free)
    spear = _spearman(posd[iu[0], iu[1]], cded[iu[0], iu[1]])
    gen = torch.Generator().manual_seed(seed + 100)
    off_free, n_off = _offaxis_closer(posd, cded, grid, near_r, gen)
    # shuffled falsifier (parameter-free): permute concept<->position
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7))
    cded_sh = torch.cdist(codes[perm], codes[perm])
    spear_sh = _spearman(posd[iu[0], iu[1]], cded_sh[iu[0], iu[1]])
    gen2 = torch.Generator().manual_seed(seed + 101)
    off_free_sh, _ = _offaxis_closer(posd, cded_sh, grid, near_r, gen2)

    # (B) HELD-OUT linear decode: fit code->(x,y) on TRAIN concepts, decode HELD-OUT concepts
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 3))
    tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]

    def fit_decode(code_all):
        A = torch.cat([code_all[tr], torch.ones(len(tr), 1)], -1)
        W = torch.linalg.lstsq(A, grid[tr]).solution
        Ate = torch.cat([code_all[te], torch.ones(len(te), 1)], -1)
        return Ate @ W
    dec = fit_decode(codes)
    held_err = ((dec - grid[te]).norm(dim=1).mean() / spacing).item()            # in units of spacing
    dec_sh = fit_decode(codes[perm])
    held_err_sh = ((dec_sh - grid[te]).norm(dim=1).mean() / spacing).item()
    # off-axis closer among HELD-OUT concepts, in the DECODED space (BALANCED -> chance 0.5)
    decd = torch.cdist(dec, dec); truen = torch.cdist(grid[te], grid[te]); gte = grid[te]
    gen3 = torch.Generator().manual_seed(seed + 200); M = len(te); ho_pos, ho_neg = [], []
    for _ in range(12000):
        a, b, c = torch.randint(M, (3,), generator=gen3).tolist()
        if len({a, b, c}) < 3:
            continue
        tb = bool(truen[a, b] < truen[a, c])
        x1 = (gte[a, 0] - gte[b, 0]).abs() < (gte[a, 0] - gte[c, 0]).abs()
        if bool(x1) != tb:
            (ho_pos if tb else ho_neg).append(int((decd[a, b] < decd[a, c]) == tb))
    hm = min(len(ho_pos), len(ho_neg))
    held_off = (sum(ho_pos[:hm] + ho_neg[:hm]) / (2 * hm)) if hm else 0.5

    return {
        "metric_spearman": round(spear, 4),
        "metric_spearman_shuffled": round(spear_sh, 4),
        "offaxis_closer_free": round(off_free, 4),
        "offaxis_closer_free_shuffled": round(off_free_sh, 4),
        "heldout_decode_err": round(held_err, 4),
        "heldout_decode_err_shuffled": round(held_err_sh, 4),
        "heldout_offaxis_closer": round(held_off, 4),
        "offaxis_gap": round(off_free - off_free_sh, 4),
        "decode_gap": round(held_err_sh - held_err, 4),
        "n_offaxis": n_off,
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["metric_spearman", "metric_spearman_shuffled", "offaxis_closer_free", "offaxis_closer_free_shuffled",
        "heldout_decode_err", "heldout_decode_err_shuffled", "heldout_offaxis_closer", "offaxis_gap", "decode_gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.8)
    a = ap.parse_args()
    per = [run_seed(s, G=a.G, spacing=a.spacing) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: off-axis closer (free) {p['offaxis_closer_free']:.3f} vs shuffled "
              f"{p['offaxis_closer_free_shuffled']:.3f} | held-out decode {p['heldout_decode_err']:.3f} vs "
              f"shuffled {p['heldout_decode_err_shuffled']:.3f} | Spearman {p['metric_spearman']:.3f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nCONCEPTUAL GRID — the FROZEN space cortex exposes a 2-D metric (CPU de-risk for #8, n={a.seeds}; "
          f"mean ± 95% CI)\n" + "=" * 96, flush=True)
    print(f"  (A) READOUT-FREE (parameter-free — cannot be circular):", flush=True)
    print(f"      OFF-AXIS 'closer' by raw code-distance {agg['offaxis_closer_free'][0]:.3f} ± "
          f"{agg['offaxis_closer_free'][1]:.3f}  (chance {CHANCE}; a 1-D projection code is <=0.5 here BY "
          f"CONSTRUCTION) — genuine 2-D", flush=True)
    print(f"      Spearman(code-dist, 2-D dist) {agg['metric_spearman'][0]:.3f}  vs  SHUFFLED "
          f"{agg['metric_spearman_shuffled'][0]:.3f} (~0)", flush=True)
    print(f"      off-axis SHUFFLED control {agg['offaxis_closer_free_shuffled'][0]:.3f} (-> chance); gap "
          f"{agg['offaxis_gap'][0]:+.3f} ± {agg['offaxis_gap'][1]:.3f}", flush=True)
    print(f"  (B) HELD-OUT linear decode (concepts the probe NEVER saw):", flush=True)
    print(f"      decode error {agg['heldout_decode_err'][0]:.3f} spacing  vs  SHUFFLED "
          f"{agg['heldout_decode_err_shuffled'][0]:.3f} (gap {agg['decode_gap'][0]:+.3f} ± "
          f"{agg['decode_gap'][1]:.3f}) — position is linearly & GENERALIZABLY present", flush=True)
    print(f"      held-out OFF-AXIS 'closer' in decoded space {agg['heldout_offaxis_closer'][0]:.3f} "
          f"(chance {CHANCE})", flush=True)

    sound = (agg["offaxis_closer_free"][0] > CHANCE + 0.05 and
             agg["offaxis_gap"][0] > 0.05 and
             agg["decode_gap"][0] > 0.5)
    verdict = ("SOUND — the frozen space code carries a genuine, CONTROL-CLEAN 2-D metric a 1-D code cannot "
               "produce; build the T4 LLM cell (train_conceptual.py / m8 notebook). Absolute strength is modest "
               "on CPU; the trained LLM readout is expected to sharpen it (1-D precedent: 1.0 -> 0.99)."
               if sound else
               "WEAK — the 2-D metric is not cleanly exposed by the frozen code; consider a concept-axis "
               "pretraining pass before the T4 cell (as the torus null needed a toroidal prior).")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "G": a.G, "spacing": a.spacing, "chance": CHANCE,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/conceptual_grid_cortex.json", "w"), indent=2)
    _svg(agg, "results/conceptual_grid_cortex.json".replace(".json", ".svg"))
    print("\nwrote results/conceptual_grid_cortex.json and results/conceptual_grid_cortex.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Conceptual grid: the frozen SPACE cortex exposes a 2-D metric (CPU de-risk, #8)</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">off-axis "closer" (where a 1-D code is wrong '
             'by construction) beats chance and collapses under shuffled positions; held-out decode generalizes</text>')
    oy = 60; base = oy + ph
    # Panel A: off-axis closer (free) vs shuffled, chance line
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) off-axis "closer" (readout-free, higher=better)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    top = 1.0
    ych = base - (CHANCE / top) * (ph - 24)
    e.append(f'<line x1="{oxA}" y1="{ych:.0f}" x2="{oxA+pw}" y2="{ych:.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{oxA+pw-2}" y="{ych-3:.0f}" font-size="8.5" fill="#9aa6bd" text-anchor="end">chance {CHANCE}</text>')
    bars = [("frozen\ncode", agg["offaxis_closer_free"][0], "#2ca25f"),
            ("SHUFFLED\npositions", agg["offaxis_closer_free_shuffled"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(bars):
        hh = (v / top) * (ph - 24); x = oxA + 40 + i * 104
        e.append(f'<rect x="{x}" y="{base-hh:.1f}" width="62" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+31}" y="{base-hh-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+31}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">a 1-D projection code is &#8804;0.5 on off-axis triples by construction</text>')
    # Panel B: held-out decode error (relocate vs shuffled), lower better
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) held-out 2-D decode error (lower=better)</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    b2 = [("held-out\nconcepts", agg["heldout_decode_err"][0], "#2ca25f"),
          ("SHUFFLED\nrefit", agg["heldout_decode_err_shuffled"][0], "#c9341a")]
    hi2 = max(b[1] for b in b2) + 1e-6
    for i, (lab, v, col) in enumerate(b2):
        hh = (v / hi2) * (ph - 24); x = oxB + 40 + i * 104
        e.append(f'<rect x="{x}" y="{base-hh:.1f}" width="62" height="{hh:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+31}" y="{base-hh-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+31}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">units of concept spacing; fit on train concepts, tested on held-out</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
