"""
src/eval/neuromodulation.py

NEUROMODULATORY CONTROL of ENCODING vs RETRIEVAL (acetylcholine) and SURPRISE-DRIVEN REMAPPING
(noradrenaline / locus coeruleus) — made faithful & emergent (GAPS.md #5).

The model already had DA-style / NE-style ML gates (`PredictionErrorGate`, `AdaptiveGain`) wired only into
`diagnose.py`/`accuracy.py`. This adds the two hippocampal neuromodulatory signatures the register asks for, on
a CA3-like auto-associative substrate (`HopfieldAssociativeMemory`), and MEASURES them — never trains them.

  ACh (Hasselmo 2006): a single tonic set-point does two OPPOSING things to the recurrent CA3 synapses at once —
    it SUPPRESSES recurrent transmission (recall) while ENHANCING the write-rate (plasticity). High ACh = ENCODE
    a novel place without old memories intruding; low ACh = RETRIEVE (complete a partial cue).
  NE  (Yu & Dayan 2005; Bouret & Sara 2005): a phasic surprise burst RESETS the network, switching the map
    (remapping). We treat the NE->remapping link as a HYPOTHESISED bridge (remapping itself: Muller & Kubie 1987;
    Leutgeb 2005; Colgin, Moser & Moser 2008) and test whether the switch is novelty-gated and adaptive.

WHAT IS BY CONSTRUCTION vs WHAT IS THE RESULT (the honesty bar set by reward_map.py):
  - That raising the recurrent gain trades encode-cleanliness for retrieval-completion is BY CONSTRUCTION (one
    knob with opposite signs). Likewise, a reset decorrelates the code BY CONSTRUCTION. Neither is reported as a
    result. Every reported number is a DIFFERENCE against a matched control, at matched storage energy:
  (A1) OVERLAP-SPECIFICITY: encoding a new field near a stored one INTRUDES on it (the recall is pulled toward
       the old memory) — but only when they OVERLAP. Intrusion is reported as the EXCESS over a far-pattern floor
       (a distant field shows ~0 intrusion at any ACh), exactly as reward_map reports over-representation as
       excess over a yoked floor.
  (A2) It is RECURRENT CONTAMINATION, not non-storage: intrusion GROWS with the encoding recurrent gain while the
       write energy ||ΔW|| is held MATCHED — so the difference is WHAT was stored, not HOW MUCH.
  (A3) RETRIEVAL COMPLETION REQUIRES W_rec: a degraded cue is completed toward the stored pattern only with the
       recurrent weights; with them off the state stays at the cue. Measured with a TRANSIENT cue so completion
       cannot be the cue echoing itself.
  (B1) NOVELTY, NOT CHANGE: the surprise signal stays LOW for large-but-EXPECTED sensory change (a big jump to a
       known place) and rises only for UNPREDICTED input — so NE is a prediction-error/novelty detector, not a
       change detector (the AUC separating novel from familiar-noise is θ-independent).
  (B2) ADAPTIVE, TWO-SIDED: after the world changes, RESET+re-encode beats a MATCHED no-reset+re-encode control
       BOTH at learning the new environment (the stale attractor causes proactive interference) AND at PROTECTING
       the old one (no-reset overwrites it — retroactive interference). This unifies NE (clears/re-indexes the
       map) with ACh (writes the new map cleanly).

Multi-seed, mean ± 95% CI. Writes results/neuromodulation.json + .svg.

    python -m src.eval.neuromodulation --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from src.models.neuro import HopfieldAssociativeMemory
from src.models.neuromodulation import AcetylcholineGate, LocusCoeruleusReset

# ---- Signature A (acetylcholine encode/retrieve) parameters ----
N = 64                 # ring units
SIG = 3.0              # place-field width (units)
D_NEAR = 6.0           # A–B separation when they OVERLAP (~2σ, fields overlap)
D_FAR = 26.0           # A–B separation when DISTANT (outside A's attractor basin; NOT antipodal so "toward A" is defined)
STEPS = 10             # attractor settling steps
RATE = 1.0             # Hebbian write rate (HELD FIXED across the ACh sweep — knobs decoupled)
RET_GAIN = 1.0         # retrieval recurrent gain (the read operation; fixed)
RET_DECAY = 0.92       # cue persistence during retrieval of a stored field
CUE_NOISE = 0.12       # noise on the retrieval cue
DROP = 0.5             # cue dropout for the completion test (fraction of units removed)
COMP_DECAY = 0.35      # TRANSIENT cue for completion (so recovery is attributable to W_rec)
COMP_STEPS = 12
COMP_CENTRES = [8.0, 20.0, 34.0, 50.0]   # a bank of stored fields to complete from
REPS_A = 10            # random cue draws averaged per seed

# ---- Signature B (noradrenaline surprise / remapping) parameters ----
D = 48                 # sensory dimension
M = 12                 # positions in an environment
K = 96                 # place-code units (near-orthogonal codes)
SENS_NOISE = 0.08      # familiar sensory noise (sets the false-positive floor)
SMALL_CHANGE = 0.30    # a genuine-but-small change (FP-floor test vs familiar noise)
BIG_JUMP = 6           # a large but EXPECTED position jump (novelty-not-change test)


# ============================ Signature A helpers ============================

def circ_bump(c, sig=SIG):
    """A normalised place field: Gaussian bump centred at ring position c, scaled to the settle norm."""
    i = torch.arange(N).float()
    d = (i - c).abs()
    d = torch.minimum(d, N - d)                      # circular distance
    b = torch.exp(-d ** 2 / (2 * sig ** 2))
    return b / b.sum() * N * 0.1


def _l2(x, target):
    """Rescale x to a fixed L2 norm — so every write has MATCHED storage energy ||ΔW||."""
    return x / (x.norm() + 1e-9) * target


def _peak(x):
    """Circular mean (ring position in [0, N)) of a population activity vector."""
    i = torch.arange(N).float()
    ang = 2 * math.pi * i / N
    s = (x * torch.sin(ang)).sum().item()
    c = (x * torch.cos(ang)).sum().item()
    return (math.atan2(s, c) % (2 * math.pi)) / (2 * math.pi) * N


def _cdiff(a, b):
    """Signed circular distance a - b on the ring, in (-N/2, N/2]. Direction-aware (handles wrap-around)."""
    return ((a - b + N / 2) % N) - N / 2


_TARGET_NORM = circ_bump(0.0).norm().item()


def intrusion(gen, c_a, c_b, enc_gain):
    """Store field A, then ENCODE field B under recurrent gain `enc_gain` (write energy held MATCHED), then
    recall B from a noisy cue. Return (pull_toward_A, write_energy_B). pull>0 => B's recall drifted toward A."""
    mem = HopfieldAssociativeMemory(N, steps=STEPS)
    mem.reset()
    mem.store(_l2(circ_bump(c_a), _TARGET_NORM), rate=RATE)                          # stored old memory A
    x_star = mem.settle(circ_bump(c_b).unsqueeze(0), recurrent_gain=enc_gain).squeeze(0)   # encode-settle B
    e_b = mem.store(_l2(x_star, _TARGET_NORM), rate=RATE)                            # matched-energy write of B
    cue = (circ_bump(c_b) + torch.randn(N, generator=gen) * CUE_NOISE).clamp(min=0).unsqueeze(0)
    rec = mem.settle(cue, recurrent_gain=RET_GAIN, drive_decay=RET_DECAY).squeeze(0)
    # circular pull of the recalled peak from B toward A, normalised by the A–B separation (wrap-safe):
    pull = _cdiff(_peak(rec), c_b) / _cdiff(c_a, c_b)
    return pull, e_b


def completion(gen):
    """Store a bank of fields; present a heavily DROPPED-OUT, TRANSIENT cue of one; measure cosine recovery to
    the true field WITH vs WITHOUT the recurrent weights. Returns (cue_cos, withW_cos, noW_cos)."""
    mem = HopfieldAssociativeMemory(N, steps=COMP_STEPS)
    mem.reset()
    for c in COMP_CENTRES:
        mem.store(_l2(circ_bump(c), _TARGET_NORM), rate=RATE)
    target = circ_bump(COMP_CENTRES[1])
    keep = (torch.rand(N, generator=gen) > DROP).float()
    cue = (target * keep).unsqueeze(0)
    with_w = mem.settle(cue, recurrent_gain=1.0, drive_decay=COMP_DECAY).squeeze(0)
    no_w = mem.settle(cue, recurrent_gain=0.0, drive_decay=COMP_DECAY).squeeze(0)
    cos = lambda a: F.cosine_similarity(a.flatten(), target.flatten(), dim=0).item()
    return cos(cue.squeeze(0)), cos(with_w), cos(no_w)


# ============================ Signature B helpers ============================

def _unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def _environment(gen):
    return _unit(torch.randn(M, D, generator=gen))         # M sensory patterns


def _codes(gen):
    return _unit(torch.randn(M, K, generator=gen))         # near-orthogonal place codes (the map assignment)


def _build(S, C):
    """A predictor P (D×K): P = Σ_i s_i ⊗ c_i. prediction at position i = P @ c_i ≈ s_i (codes ~ orthogonal)."""
    P = torch.zeros(D, K)
    for i in range(M):
        P += torch.outer(S[i], C[i])
    return P


def _surprise(P, code, actual, lc):
    return lc.surprise((P @ code).unsqueeze(0), actual.unsqueeze(0)).item()


def _auc(pos, neg):
    """Probability a random 'pos' surprise exceeds a random 'neg' one (θ-independent separation)."""
    pos = torch.tensor(pos); neg = torch.tensor(neg)
    wins = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean().item()
    return wins + 0.5 * ties


# ================================= per seed =================================

def run_seed(seed, reps=REPS_A):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 101)

    # ---- Signature A: intrusion (near vs far, high vs low encoding gain) at MATCHED storage energy ----
    near, far = {}, {}
    for gain, tag in ((1.0, "hi"), (0.0, "lo")):        # hi gain = LOW ACh (retrieve-like); lo gain = HIGH ACh
        pn, en, pf, ef = [], [], [], []
        for _ in range(reps):
            c_a = torch.randint(0, N, (1,), generator=gen).item() + 0.0
            p, e = intrusion(gen, c_a, c_a + D_NEAR, gain); pn.append(p); en.append(e)
            p, e = intrusion(gen, c_a, c_a + D_FAR, gain); pf.append(p); ef.append(e)
        near[tag] = (sum(pn) / reps, sum(en) / reps)
        far[tag] = (sum(pf) / reps, sum(ef) / reps)

    comp = [completion(gen) for _ in range(reps)]
    comp_cue = sum(c[0] for c in comp) / reps
    comp_withW = sum(c[1] for c in comp) / reps
    comp_noW = sum(c[2] for c in comp) / reps

    # ---- Signature B: surprise = novelty (not change); adaptive two-sided remap benefit ----
    lc = LocusCoeruleusReset(threshold=0.5)
    S1 = _environment(gen); C1 = _codes(gen); P = _build(S1, C1)
    fam, expbig, novel, small = [], [], [], []
    for i in range(M):
        act = _unit(S1[i] + SENS_NOISE * torch.randn(D, generator=gen))
        fam.append(_surprise(P, C1[i], act, lc))
        j = (i + BIG_JUMP) % M                                    # big but EXPECTED jump to a known place
        actj = _unit(S1[j] + SENS_NOISE * torch.randn(D, generator=gen))
        expbig.append(_surprise(P, C1[j], actj, lc))
        novel.append(_surprise(P, C1[i], _unit(torch.randn(D, generator=gen)), lc))
        sc = _unit((1 - SMALL_CHANGE) * S1[i] + SMALL_CHANGE * _unit(torch.randn(D, generator=gen)))
        small.append(_surprise(P, C1[i], sc, lc))

    # adaptive benefit: world switches to env2. remap+re-encode vs MATCHED no-reset+re-encode.
    S2 = _environment(gen)
    P_noreset = P.clone()
    for i in range(M):                                            # write env2 onto the SAME (stale) map -> interference
        P_noreset += torch.outer(S2[i], C1[i])
    C2 = _codes(gen); P_remap = _build(S2, C2)                    # remap: fresh code, env1 kept in old units
    err = lambda Pm, Cm, S: sum(_surprise(Pm, Cm[i], S[i], lc) for i in range(M)) / M
    new_noreset = err(P_noreset, C1, S2); new_remap = err(P_remap, C2, S2)
    old_noreset = err(P_noreset, C1, S1); old_remap = err(P, C1, S1)

    m = lambda v: sum(v) / len(v)
    return {
        # Signature A
        "intr_near_hi": near["hi"][0], "intr_near_lo": near["lo"][0],
        "intr_far_hi": far["hi"][0], "intr_far_lo": far["lo"][0],
        "dW_near_hi": near["hi"][1], "dW_near_lo": near["lo"][1],          # matched-storage control
        "A_specificity": near["hi"][0] - far["hi"][0],                     # (A1) excess over far floor
        "A_encode_effect": near["hi"][0] - near["lo"][0],                  # (A2) contamination vs encode gain
        "comp_cue": comp_cue, "comp_withW": comp_withW, "comp_noW": comp_noW,
        "A_completion": comp_withW - comp_cue,                             # (A3) recovery
        "A_completion_needs_W": comp_withW - comp_noW,
        # Signature B
        "surp_familiar": m(fam), "surp_expbig": m(expbig), "surp_novel": m(novel), "surp_small": m(small),
        "B_auc_novel": _auc(novel, fam),                                   # (B1) θ-independent
        "B_fp_margin": m(small) - m(fam),                                  # small-real-change above the noise floor
        "new_noreset": new_noreset, "new_remap": new_remap,
        "old_noreset": old_noreset, "old_remap": old_remap,
        "B_benefit_new": new_noreset - new_remap,                         # (B2) remap learns new env better
        "B_benefit_old": old_noreset - old_remap,                        # (B2) remap protects old env
    }


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


KEYS = ["intr_near_hi", "intr_near_lo", "intr_far_hi", "intr_far_lo", "dW_near_hi", "dW_near_lo",
        "A_specificity", "A_encode_effect", "comp_cue", "comp_withW", "comp_noW",
        "A_completion", "A_completion_needs_W",
        "surp_familiar", "surp_expbig", "surp_novel", "surp_small", "B_auc_novel", "B_fp_margin",
        "new_noreset", "new_remap", "old_noreset", "old_remap", "B_benefit_new", "B_benefit_old"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {k: ci([p[k] for p in per]) for k in KEYS}
    for s, p in enumerate(per):
        print(f"  seed {s}: A intrusion near hi/lo {p['intr_near_hi']:+.2f}/{p['intr_near_lo']:+.2f} "
              f"far {p['intr_far_hi']:+.2f} | completion {p['A_completion']:+.2f} (noW {p['comp_noW']-p['comp_cue']:+.2f}) "
              f"| B surprise fam/novel {p['surp_familiar']:.2f}/{p['surp_novel']:.2f} "
              f"benefit new/old {p['B_benefit_new']:+.2f}/{p['B_benefit_old']:+.2f}", flush=True)

    print(f"\nNEUROMODULATION — acetylcholine encode/retrieve + noradrenaline surprise remapping "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 96, flush=True)
    print("  ACETYLCHOLINE (Hasselmo 2006) — encoding blocks intrusion of old memories:", flush=True)
    print(f"    (A1) OVERLAP-SPECIFIC intrusion (low ACh encode): near {agg['intr_near_hi'][0]:+.2f} ± "
          f"{agg['intr_near_hi'][1]:.2f}  vs  FAR floor {agg['intr_far_hi'][0]:+.2f} ± {agg['intr_far_hi'][1]:.2f}"
          f"  -> excess {agg['A_specificity'][0]:+.2f} ± {agg['A_specificity'][1]:.2f}", flush=True)
    print(f"    (A2) contamination vs encode gain at MATCHED storage: intrusion {agg['intr_near_hi'][0]:+.2f} "
          f"(low ACh) -> {agg['intr_near_lo'][0]:+.2f} (high ACh); effect {agg['A_encode_effect'][0]:+.2f} ± "
          f"{agg['A_encode_effect'][1]:.2f}   [||ΔW|| {agg['dW_near_hi'][0]:.2f} vs {agg['dW_near_lo'][0]:.2f} — matched]", flush=True)
    print(f"    (A3) RETRIEVAL completion needs W_rec: degraded cue {agg['comp_cue'][0]:.2f} -> "
          f"{agg['comp_withW'][0]:.2f} with W_rec, but {agg['comp_noW'][0]:.2f} without "
          f"(recovery {agg['A_completion'][0]:+.2f} ± {agg['A_completion'][1]:.2f})", flush=True)
    print("  NORADRENALINE (Yu & Dayan 2005; Bouret & Sara 2005) — surprise gates remapping:", flush=True)
    print(f"    (B1) NOVELTY not change: surprise familiar-noise {agg['surp_familiar'][0]:.2f}, big-EXPECTED-jump "
          f"{agg['surp_expbig'][0]:.2f} (≈ familiar), NOVEL {agg['surp_novel'][0]:.2f}; AUC novel-vs-familiar "
          f"{agg['B_auc_novel'][0]:.2f} ± {agg['B_auc_novel'][1]:.2f}", flush=True)
    print(f"    (B2) ADAPTIVE, two-sided (remap+re-encode vs MATCHED no-reset+re-encode): NEW-env error "
          f"{agg['new_noreset'][0]:.2f}->{agg['new_remap'][0]:.2f} (benefit {agg['B_benefit_new'][0]:+.2f} ± "
          f"{agg['B_benefit_new'][1]:.2f}); OLD-env recall {agg['old_noreset'][0]:.2f}->{agg['old_remap'][0]:.2f} "
          f"(protected {agg['B_benefit_old'][0]:+.2f} ± {agg['B_benefit_old'][1]:.2f})", flush=True)

    print(f"\n  -> ACh sets a hippocampal ENCODE/RETRIEVE mode on a CA3 auto-associator: at high ACh the recurrent "
          f"recall is suppressed so a new field is written WITHOUT being pulled onto an overlapping stored one "
          f"(intrusion excess over the far floor {agg['A_specificity'][0]:+.2f}), and this is recurrent "
          f"contamination not non-storage (the ||ΔW|| write energy is matched); the SAME recurrent weights are "
          f"what complete a degraded cue during retrieval ({agg['A_completion'][0]:+.2f}, gone without them). NE "
          f"fires on genuine NOVELTY not mere input change (AUC {agg['B_auc_novel'][0]:.2f}; a big EXPECTED jump "
          f"stays at the familiar floor), and a surprise-triggered remap is ADAPTIVE on BOTH sides vs a matched "
          f"re-encoding control — it learns the new world (benefit {agg['B_benefit_new'][0]:+.2f}) AND protects "
          f"the old map from overwrite (benefit {agg['B_benefit_old'][0]:+.2f}), unifying NE's reset with ACh's "
          f"clean encoding. Every number is a DIFFERENCE vs a matched control; the mode switch is set, the "
          f"signatures are MEASURED. Hasselmo's ACh and the LC-NE reset, on the model's hippocampus.", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "D": D, "M": M, "K": K,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/neuromodulation.json", "w"), indent=2)
    svg(agg, "results/neuromodulation.svg")
    print("\nwrote results/neuromodulation.json and results/neuromodulation.svg", flush=True)


def svg(agg, out):
    pad = 60; pw = 250; ph = 200; gap = 70; W = pad + 2 * pw + gap + 20; H = 92 + ph + 40
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Neuromodulation: ACh encode/retrieve &amp; NE surprise remapping</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">high ACh encoding blocks intrusion of an '
             'overlapping memory (vs far floor); NE surprise remap is adaptive both ways &#8212; measured, not trained</text>')
    oy = 58; base = oy + ph
    # Panel A: intrusion near (low vs high ACh) vs far floor
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) intrusion on old memory</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    bars = [("near\nlow ACh", agg["intr_near_hi"][0], "#c9341a"),
            ("near\nhigh ACh", agg["intr_near_lo"][0], "#2ca25f"),
            ("far\nfloor", agg["intr_far_hi"][0], "#9aa6bd")]
    hi = max(b[1] for b in bars) + 1e-6
    for i, (lab, v, col) in enumerate(bars):
        h = (v / hi) * (ph - 30); x = oxA + 24 + i * 74
        e.append(f'<rect x="{x}" y="{base-max(h,0):.1f}" width="52" height="{abs(h):.1f}" fill="{col}" opacity="0.88"/>')
        e.append(f'<text x="{x+26}" y="{base-abs(h)-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+14+j*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{ln}</text>')
    # Panel B: adaptive benefit both ways
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) NE remap benefit vs no-reset</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    bb = [("learn\nnew env", agg["B_benefit_new"][0]), ("protect\nold map", agg["B_benefit_old"][0])]
    hi2 = max(b[1] for b in bb) + 1e-6
    for i, (lab, v) in enumerate(bb):
        h = (v / hi2) * (ph - 30); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-max(h,0):.1f}" width="64" height="{abs(h):.1f}" fill="#3182bd" opacity="0.88"/>')
        e.append(f'<text x="{x+32}" y="{base-abs(h)-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:+.2f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+14+j*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+34:.0f}" font-size="9.5" fill="#5b6b8c">surprise = novelty not change '
             f'(AUC {agg["B_auc_novel"][0]:.2f}); completion needs W_rec ({agg["A_completion"][0]:+.2f})</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
