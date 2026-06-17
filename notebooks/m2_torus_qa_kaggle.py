# Milestone 2 — TORUS-QA: the leakage-proof CAUSAL headline (Kaggle T4)
# ---------------------------------------------------------------------------
# A frozen cortex lets Qwen answer navigation questions on a world with NO faithful Euclidean text
# description: a TORUS (the board wraps at its edges). True position = (integral of self-motion) mod L,
# so the answer (which of 9 wrap-around cells) requires path integration WITH wrap — a language prior
# over Euclidean space cannot supply it, and the moves never appear in the prompt. The headline is the
# CAUSAL + leakage-proof contrast: cortex-ON >> cortex-OFF (text-only, ~chance 11%).
#
# Design validated on CPU first (src/eval/torus_qa.py): a small readout on the grid cortex solves the
# toroidal cells (76-99%) while a text-only proxy is at chance (11%). This cell runs it through Qwen.
#
# IMPORTANT FIX (after a first null run): the cortex is pre-trained self-supervised and FROZEN, and a
# Euclidean-pretrained readout HIDES the toroidal cell from the LLM (frozen probe = chance 16%). The
# trainer now pre-trains a TOROIDAL self-supervised target for --task torus (harmonics of L; place/grid
# codes are environment-specific — a real toroidal grid manifold, Gardner 2022). Faithful CPU check:
# frozen toroidally-pretrained cortex -> 85-98% on torus cells. If you ran the pre-fix version, DELETE
# results/torus_llm/*.json before re-running cell 3.
# RESUMABLE: skips runs already written. Copy each block into its own Kaggle cell. ~30-60 min/run on T4.
# ===========================================================================


# %% [cell 1] setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
!pip -q uninstall -y torchao
print("setup done")


# %% [cell 2] cache the base LLM
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] torus-QA sweep (resumable). grid (constrained_velocity) vs place/default cortex x seeds.
# Each run auto-evaluates cortex ON vs OFF per length -> the leakage-proof causal contrast.
import os, subprocess, time
SEEDS = [0, 1, 2]
OUTDIR = "results/torus_llm"; os.makedirs(OUTDIR, exist_ok=True)
CORTEX = {"grid": ["--constrained_velocity"], "place": []}
COMMON = ["--task", "torus", "--epochs", "5", "--early_stop", "--n_train", "2400", "--n_val", "300",
          "--train_lengths", "6", "8", "10", "12", "--eval_lengths", "8", "16", "24"]
jobs = [(c, s) for c in CORTEX for s in SEEDS]
print(f"{len(jobs)} runs (done ones skipped)\n")
for cortex, seed in jobs:
    out = f"{OUTDIR}/torus_{cortex}_s{seed}.json"
    if os.path.exists(out):
        print(f"skip  torus {cortex:5} seed={seed}"); continue
    cmd = (["python", "-u", "-m", "src.training.train_trajectory"] + COMMON
           + ["--seed", str(seed), "--out", out] + CORTEX[cortex])
    print(f"\n>>> torus {cortex} seed={seed}", flush=True); t0 = time.time()
    subprocess.run(cmd, check=True); print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\ntorus sweep pass complete (re-run if any remain)")


# %% [cell 4] aggregate: cortex ON vs OFF (the headline) + grid vs place, with paired tests
import os, json, math, random
OUTDIR = "results/torus_llm"; SEEDS = list(range(8)); LENS = [8, 16, 24]; CHANCE = 1/9


def load(cortex, s):
    f = f"{OUTDIR}/torus_{cortex}_s{s}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def ci95(xs):
    n = len(xs)
    if n == 0: return None
    m = sum(xs)/n; sd = (sum((x-m)**2 for x in xs)/(n-1))**0.5 if n > 1 else 0.0
    return m, 1.96*sd/math.sqrt(n)


def paired_p(diffs, iters=20000):
    n = len(diffs); m = sum(diffs)/n; rng = random.Random(0)
    return sum(abs(sum(d*(1 if rng.random()<0.5 else -1) for d in diffs)/n) >= abs(m)-1e-12 for _ in range(iters))/iters


agg = {}
for cortex in ("grid", "place"):
    rows = [load(cortex, s) for s in SEEDS if load(cortex, s)]
    if not rows: continue
    print("=" * 70); print(f"CORTEX = {cortex}   (n={len(rows)} seeds; exact cell accuracy, chance={CHANCE:.0%})")
    print("  T      cortex-ON          cortex-OFF        Delta(ON-OFF)  p")
    agg[cortex] = {}
    for T in LENS:
        on = [r["results_by_len"][str(T)]["cortex_on_exact"] for r in rows if str(T) in r["results_by_len"]]
        off = [r["results_by_len"][str(T)]["cortex_off_exact"] for r in rows if str(T) in r["results_by_len"]]
        n = min(len(on), len(off))
        mo, co = ci95(on); mf, cf = ci95(off)
        d = [on[i]-off[i] for i in range(n)]; p = paired_p(d) if n >= 2 else float("nan")
        agg[cortex][T] = {"on": [round(mo,4), round(co,4)], "off": [round(mf,4), round(cf,4)],
                          "delta": round(sum(d)/n,4), "p": round(p,4), "n": n}
        print(f"  T={T:<3}  {mo:.0%} +/-{co:.0%}      {mf:.0%} +/-{cf:.0%}     {sum(d)/n:+.0%}        {p:.4f}")
json.dump({"chance": CHANCE, "eval_lengths": LENS, "aggregate": agg},
          open("results/torus_llm.json", "w"), indent=2)
print("\nwrote results/torus_llm.json  <-- paste the tables above back, or share this file")
print("Headline: cortex-ON >> cortex-OFF (chance) on a torus = the LLM reasons through path integration,")
print("on a world with no Euclidean text prior (leakage-proof).")
