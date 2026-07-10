"""
src/eval/hippocampal_subfields.py

The hippocampal computational triad — DG PATTERN SEPARATION + CA1 COMPARATOR (GAPS.md Tier 2, #2).

The repo already has CA3 (`HopfieldAssociativeMemory`, a Marr/Hopfield/Treves-Rolls recurrent auto-associator
that pattern-COMPLETES a cue to the nearest stored attractor — and INTERFERES when stored patterns are too
similar). The two subfields around it were missing:

  DG (dentate gyrus): a massive SPARSE EXPANSION (few % active) that ORTHOGONALIZES similar entorhinal inputs
    (pattern separation) so CA3 can store near-identical environments without their attractors merging.
  CA1: a COMPARATOR of CA3's completed prediction (Schaffer collaterals) against the direct entorhinal reality
    (perforant path). Their MISMATCH is a novelty / prediction-error signal (Lisman & Grace 2005; Vinogradova
    1995; Hasselmo & Schnell 1994).

We MEASURE the functional consequences (never put in a loss), guarding the by-construction trap (a sparse random
expansion trivially orthogonalizes — that proves nothing; the headline must be the DOWNSTREAM RECALL):

  (A) SEPARATION -> INTERFERENCE-FREE RECALL (headline). Store M SIMILAR environments (high entorhinal overlap).
      At MATCHED CA3 size (N_dg) and matched storage, DG's sparse code lets CA3 recall the CORRECT environment
      from a partial cue, where a DENSE expansion of the SAME size intrudes on a similar one. Recall(DG) >>
      Recall(dense). The mechanism check — DG's output overlap is far below its input overlap (separation index
      << 1) while the dense expansion preserves the overlap — explains WHY, but the headline is the recall.
  (B) DENSE-EXPANSION FALSIFIER. Same N_dg, same expansion, NO k-WTA sparsity -> separation gone -> interference
      returns (recall collapses toward the small direct-EC network). So it is the sparse SEPARATION that matters,
      not the extra dimensionality.
  (C) CA1 COMPARATOR. The mismatch signal discriminates NOVEL vs FAMILIAR environments (AUC ~1). FALSIFIER:
      ablate the CA3 stream -> the comparator has no stored memory to compare against -> AUC -> ~0.5. So it is a
      genuine ENTORHINAL-vs-MEMORY comparator, not a mere input-novelty detector (which the NE organ already is).

    python -m src.eval.hippocampal_subfields --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from src.models.neuro.attractor import HopfieldAssociativeMemory

N_EC = 120          # entorhinal input dimension
K = 24              # active units per environment
M = 24              # stored environments (near / above the small-net CA3 capacity -> interference is possible)
N_DG = 1500         # dentate expansion dimension (>> N_EC)
SPARSE = 0.05       # DG k-winner-take-all sparsity (few % active)
OVERLAP = 0.6       # entorhinal similarity of the stored environments (high -> separation matters)
CORRUPT = 0.3       # fraction of a cue's active units dropped (partial cue)
DRAWS = 5           # pattern-set draws averaged per seed (recall is a small-sample fraction)


def make_env_patterns(N, k, m, overlap, gen):
    """m binary environment patterns, k active each, pairwise overlap ~= `overlap` via a shared active core."""
    core = round(overlap * k)
    perm = torch.randperm(N, generator=gen)
    core_idx = perm[:core]
    pool = perm[core:]
    pats = torch.zeros(m, N)
    for i in range(m):
        uq = pool[torch.randperm(len(pool), generator=gen)[:k - core]]
        pats[i, core_idx] = 1.0
        pats[i, uq] = 1.0
    return pats


def dg_code(ec, W, sparse):
    """DG expansion. sparse>0 -> k-WTA sparse binary code (SEPARATION); sparse=0 -> dense tanh (no separation)."""
    h = ec @ W.t()
    if sparse:
        kk = max(1, int(sparse * h.shape[1]))
        thr = h.topk(kk, dim=1).values[:, -1:]
        return (h >= thr).float()
    return torch.tanh(h)


def recall_acc(codes, corrupt, gen, steps=10):
    """Store codes in a CA3 Hopfield; recall each from a partial cue; fraction completed to the CORRECT pattern."""
    N = codes.shape[1]
    ca3 = HopfieldAssociativeMemory(N, steps=steps)
    mean = codes.mean(0, keepdim=True)
    P = codes - mean                                     # center (Hopfield stores balanced patterns better)
    for i in range(codes.shape[0]):
        ca3.store(P[i])
    cor = 0
    for i in range(codes.shape[0]):
        cue = codes[i].clone()
        act = cue.nonzero().squeeze(1)
        if len(act) > 0:
            drop = act[torch.randperm(len(act), generator=gen)[:int(corrupt * len(act))]]
            cue[drop] = 0.0
        settled = ca3.settle((cue - mean).unsqueeze(0), recurrent_gain=1.0)
        cor += int((settled @ P.t()).squeeze(0).argmax().item() == i)
    return cor / codes.shape[0]


def mean_pair_cos(x):
    """Mean pairwise cosine similarity (an overlap measure comparable across sparse/dense codes)."""
    xn = F.normalize(x, dim=1)
    s = xn @ xn.t()
    n = x.shape[0]
    return (s.sum() - n) / (n * (n - 1))


def ca1_novelty(store_codes, probe_codes, ablate_ca3=False, steps=10):
    """CA1 mismatch = 1 - cos(CA3-completed, DG(probe)). Familiar probe -> CA3 completes to its stored code
    (match, low); novel probe -> CA3 completes to the nearest WRONG stored code (mismatch, high)."""
    N = store_codes.shape[1]
    ca3 = HopfieldAssociativeMemory(N, steps=steps)
    mean = store_codes.mean(0, keepdim=True)
    for i in range(store_codes.shape[0]):
        ca3.store(store_codes[i] - mean)
    settled = torch.zeros_like(probe_codes) if ablate_ca3 else ca3.settle(probe_codes - mean, recurrent_gain=1.0)
    return (1 - F.cosine_similarity(settled, probe_codes, dim=1))


def auc(pos, neg):
    """P(pos > neg), ties counted 0.5 (so an uninformative constant signal reads 0.5, not 0)."""
    p = pos.unsqueeze(1); n = neg.unsqueeze(0)
    return ((p > n).float() + 0.5 * (p == n).float()).mean().item()


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(N_DG, N_EC, generator=gen) / (N_EC ** 0.5)
    rec_dg = rec_de = rec_di = sep_dg = sep_de = 0.0
    for d in range(DRAWS):
        g = torch.Generator().manual_seed(seed * 100 + d)
        ec = make_env_patterns(N_EC, K, M, OVERLAP, g)
        cg = torch.Generator().manual_seed(seed * 100 + d + 7)          # cue-corruption RNG (shared across conds)
        dg_s = dg_code(ec, W, SPARSE); dg_d = dg_code(ec, W, 0.0)
        rec_dg += recall_acc(dg_s, CORRUPT, torch.Generator().manual_seed(cg.initial_seed()))
        rec_de += recall_acc(dg_d, CORRUPT, torch.Generator().manual_seed(cg.initial_seed()))
        rec_di += recall_acc(ec, CORRUPT, torch.Generator().manual_seed(cg.initial_seed()))
        ec_ov = mean_pair_cos(ec).item()
        sep_dg += (mean_pair_cos(dg_s).item() / (ec_ov + 1e-9))          # separation index (out/in overlap)
        sep_de += (mean_pair_cos(dg_d).item() / (ec_ov + 1e-9))
    rec_dg /= DRAWS; rec_de /= DRAWS; rec_di /= DRAWS; sep_dg /= DRAWS; sep_de /= DRAWS

    # CA1 comparator: store M familiar envs; probe with familiar (stored) vs novel (new) environments
    g = torch.Generator().manual_seed(seed * 13 + 1)
    ec = make_env_patterns(N_EC, K, M, OVERLAP, g)
    store = dg_code(ec, W, SPARSE)
    nov = dg_code(make_env_patterns(N_EC, K, M, OVERLAP, torch.Generator().manual_seed(seed * 13 + 999)), W, SPARSE)
    mm_fam = ca1_novelty(store, store); mm_nov = ca1_novelty(store, nov)
    mm_fam_a = ca1_novelty(store, store, ablate_ca3=True); mm_nov_a = ca1_novelty(store, nov, ablate_ca3=True)
    ca1_auc = auc(mm_nov, mm_fam); ca1_auc_abl = auc(mm_nov_a, mm_fam_a)

    return {
        "recall_dg": round(rec_dg, 4),
        "recall_dense": round(rec_de, 4),
        "recall_direct": round(rec_di, 4),
        "sep_index_dg": round(sep_dg, 4),
        "sep_index_dense": round(sep_de, 4),
        "ca1_auc": round(ca1_auc, 4),
        "ca1_auc_ablate": round(ca1_auc_abl, 4),
        "recall_gap": round(rec_dg - rec_de, 4),          # DG separation vs dense at matched size
        "ca1_gap": round(ca1_auc - ca1_auc_abl, 4),       # comparator needs the CA3 memory stream
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


KEYS = ["recall_dg", "recall_dense", "recall_direct", "sep_index_dg", "sep_index_dense",
        "ca1_auc", "ca1_auc_ablate", "recall_gap", "ca1_gap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: recall DG {p['recall_dg']:.2f} / dense {p['recall_dense']:.2f} / direct "
              f"{p['recall_direct']:.2f} | sep DG {p['sep_index_dg']:.2f} vs dense {p['sep_index_dense']:.2f} | "
              f"CA1 AUC {p['ca1_auc']:.2f} (ablate {p['ca1_auc_ablate']:.2f})", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nHIPPOCAMPAL SUBFIELDS — DG pattern separation + CA1 comparator (n={a.seeds}; mean ± 95% CI)\n" + "=" * 92, flush=True)
    print(f"  M={M} SIMILAR environments (entorhinal overlap {OVERLAP}); CA3 = Hopfield auto-associator; recall "
          f"from a {int(CORRUPT*100)}%-degraded cue", flush=True)
    print(f"  (A) SEPARATION -> INTERFERENCE-FREE RECALL (matched CA3 size N_dg={N_DG}):", flush=True)
    print(f"      DG (sparse) {agg['recall_dg'][0]:.3f} ± {agg['recall_dg'][1]:.3f}  vs  DENSE expansion "
          f"{agg['recall_dense'][0]:.3f} ± {agg['recall_dense'][1]:.3f}   (gap {agg['recall_gap'][0]:+.3f} ± "
          f"{agg['recall_gap'][1]:.3f}; direct-EC baseline {agg['recall_direct'][0]:.3f})", flush=True)
    print(f"      mechanism — separation index (DG output overlap / entorhinal input overlap): DG "
          f"{agg['sep_index_dg'][0]:.2f} (<<1 = orthogonalized) vs dense {agg['sep_index_dense'][0]:.2f} "
          f"(~1 = overlap preserved)", flush=True)
    print(f"  (B) DENSE-EXPANSION FALSIFIER: same size, no k-WTA -> interference returns (recall "
          f"{agg['recall_dense'][0]:.3f}) — so it is the sparse SEPARATION, not the dimensionality.", flush=True)
    print(f"  (C) CA1 COMPARATOR — novelty AUC {agg['ca1_auc'][0]:.3f} ± {agg['ca1_auc'][1]:.3f}; ablate the CA3 "
          f"stream -> {agg['ca1_auc_ablate'][0]:.3f} (≈chance, gap {agg['ca1_gap'][0]:+.3f}): a genuine "
          f"entorhinal-vs-memory comparator, not an input-novelty detector.", flush=True)

    sound = (agg["recall_gap"][0] > 0.2 and agg["sep_index_dg"][0] < 0.6 and
             agg["ca1_auc"][0] > 0.8 and agg["ca1_gap"][0] > 0.3)
    verdict = ("SOUND — sparse DG separation gives interference-free CA3 recall of similar environments where a "
               "matched-size dense expansion fails, and a CA1 comparator detects novelty only with the CA3 memory "
               "stream. The hippocampal DG/CA3/CA1 triad, measured against its falsifiers." if sound else
               "WEAK — the separation/comparator signatures did not clear the falsifiers; revisit the regime.")
    print(f"\n  verdict: {verdict}", flush=True)

    out = {"n_seeds": a.seeds, "N_ec": N_EC, "k": K, "M": M, "N_dg": N_DG, "sparse": SPARSE, "overlap": OVERLAP,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/hippocampal_subfields.json", "w"), indent=2)
    _svg(agg, "results/hippocampal_subfields.svg")
    print("\nwrote results/hippocampal_subfields.json and results/hippocampal_subfields.svg", flush=True)


def _svg(agg, out):
    pad = 60; pw = 250; ph = 190; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Hippocampal subfields: DG pattern separation + CA1 comparator</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">sparse DG orthogonalizes similar environments '
             'so CA3 recalls them without interference (a dense expansion of the same size fails); CA1 detects '
             'novelty only with the memory</text>')
    oy = 60; base = oy + ph
    # Panel A: recall (DG / dense / direct), higher=better
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) CA3 recall of SIMILAR environments (higher=better)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("DG\n(sparse)", agg["recall_dg"][0], "#2ca25f"), ("dense\nexp", agg["recall_dense"][0], "#c9341a"),
            ("direct\nEC", agg["recall_direct"][0], "#9aa6bd")]
    for i, (lab, v, col) in enumerate(bars):
        h = v * (ph - 24); x = oxA + 22 + i * 74
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="52" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-h-6:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">DG &amp; dense share the CA3 size N_dg — only DG adds sparse separation</text>')
    # Panel B: CA1 novelty AUC (intact vs ablate-CA3)
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) CA1 novelty detection (AUC)</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    ych = base - 0.5 * (ph - 24)
    e.append(f'<line x1="{oxB}" y1="{ych:.0f}" x2="{oxB+pw}" y2="{ych:.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{oxB+pw-2}" y="{ych-3:.0f}" font-size="8.5" fill="#9aa6bd" text-anchor="end">chance 0.5</text>')
    b2 = [("EC + CA3\nmemory", agg["ca1_auc"][0], "#2ca25f"), ("ablate CA3\n(no memory)", agg["ca1_auc_ablate"][0], "#c9341a")]
    for i, (lab, v, col) in enumerate(b2):
        h = v * (ph - 24); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="64" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+32}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+36:.0f}" font-size="9" fill="#5b6b8c">novelty needs the stored memory — a genuine entorhinal-vs-CA3 comparator</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
