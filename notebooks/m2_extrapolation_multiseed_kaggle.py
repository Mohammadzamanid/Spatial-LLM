# Milestone 2 — Figure 2: LLM-level length extrapolation, MULTI-SEED with 95% CI (Kaggle T4)
# ---------------------------------------------------------------------------
# The central claim, at the language level, made publication-rigorous: does the velocity-driven
# GRID-CELL cortex let Qwen answer about paths LONGER than training, with error bars, beating the
# place/default cortex? We sweep {grid, place} x {distance, bearing} x seeds and aggregate the
# per-length cortex-ON accuracy into mean +/- 95% CI. (Figure 1 — the representation alone, no LLM —
# is results/extrapolation.svg; this is its language-level counterpart.)
#
# grid cortex  = --constrained_velocity (velocity-driven hexagonal grid modules; README section 4)
# place/default= no flag (bounded place / attractor cortex)
# cortex-OFF   = text-only control, reported automatically (~chance) -> proves the answer rides on the code
#
# RESUMABLE: each (task, cortex, seed) writes its own JSON and is SKIPPED if already present, so you
# can run this across several T4 sessions (the full matrix exceeds one 9 h session). DISTANCE (the
# magnitude discriminator) is ordered first — it is the result that matters most. Copy each block
# into its own Kaggle cell.
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


# %% [cell 3] the multi-seed sweep (resumable; re-run across sessions until everything exists)
import os, subprocess, time

SEEDS = [0, 1, 2]                       # 3 = minimum for a CI; bump to 5 if you have the hours
OUTDIR = "results/extrap_llm"
os.makedirs(OUTDIR, exist_ok=True)

# per-task trainer flags. distance is unstable -> early-stop + low LR + more epochs (locks best-val).
TASKS = {
    "distance": ["--epochs", "8", "--lr", "1e-4", "--early_stop"],     # the magnitude discriminator
    "bearing":  ["--epochs", "3"],                                     # scale-free direction
}
CORTEX = {"grid": ["--constrained_velocity"], "place": []}             # grid vs place/default

COMMON = ["--n_train", "2400", "--n_val", "300",
          "--train_lengths", "6", "8", "10", "12", "--eval_lengths", "8", "16", "24"]

jobs = [(t, c, s) for t in TASKS for c in CORTEX for s in SEEDS]       # distance first (dict order)
print(f"{len(jobs)} runs total; already-done ones are skipped\n")
for task, cortex, seed in jobs:
    out = f"{OUTDIR}/{task}_{cortex}_s{seed}.json"
    if os.path.exists(out):
        print(f"skip  {task:8} {cortex:5} seed={seed}  (exists)"); continue
    cmd = (["python", "-u", "-m", "src.training.train_trajectory", "--task", task, "--seed", str(seed),
            "--out", out] + CORTEX[cortex] + TASKS[task] + COMMON)
    print(f"\n>>> {task} {cortex} seed={seed}\n    " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nsweep pass complete (re-run this cell if any runs remain)")


# %% [cell 4] aggregate -> mean +/- 95% CI (paste this output back; it also writes an aggregate JSON)
import os, json, math

OUTDIR = "results/extrap_llm"
SEEDS = [0, 1, 2]
TASKS = ["distance", "bearing"]
CORTEX = ["grid", "place"]
LENS = [8, 16, 24]


def ci95(xs):
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return round(m, 4), round(1.96 * sd / math.sqrt(n), 4)


agg = {}
for task in TASKS:
    print("=" * 78); print(f"TASK = {task}   (cortex-ON exact accuracy; mean +/- 95% CI over seeds)")
    print("  cortex   " + "".join(f"T={T}".rjust(18) for T in LENS) + "     n  | OFF@T24")
    agg[task] = {}
    for cortex in CORTEX:
        rows = []
        for s in SEEDS:
            f = f"{OUTDIR}/{task}_{cortex}_s{s}.json"
            if os.path.exists(f):
                rows.append(json.load(open(f)))
        agg[task][cortex] = {"n": len(rows), "by_len": {}}
        cells = []
        for T in LENS:
            on = [r["results_by_len"][str(T)]["cortex_on_exact"] for r in rows
                  if str(T) in r["results_by_len"]]
            c = ci95(on)
            agg[task][cortex]["by_len"][T] = ({"mean": c[0], "ci95": c[1]} if c else None)
            cells.append((f"{c[0]:.0%} +/-{c[1]:.0%}" if c else "--").rjust(18))
        off24 = ci95([r["results_by_len"]["24"]["cortex_off_exact"] for r in rows
                      if "24" in r["results_by_len"]])
        agg[task][cortex]["off_T24"] = ({"mean": off24[0], "ci95": off24[1]} if off24 else None)
        offs = f"{off24[0]:.0%}" if off24 else "--"
        print(f"  {cortex:7}" + "".join(cells) + f"   {len(rows):3d}  | {offs}")

with open("results/extrapolation_llm.json", "w") as fh:
    json.dump({"seeds": SEEDS, "tasks": TASKS, "cortex": CORTEX, "eval_lengths": LENS,
               "metric": "cortex_on_exact", "aggregate": agg}, fh, indent=2)
print("\nwrote results/extrapolation_llm.json  <-- paste the table above back, or share this file")


# %% [cell 5] BEARING-only, n=8 (the one worthwhile follow-up: resolve grid-vs-place on the
# trending, stable task). distance is too seed-variable at this scale to resolve, so we DON'T spend
# T4 hours on it. Resumable: skips runs already done (incl. the 3 bearing seeds from cell 3).
import os, subprocess, time
OUTDIR = "results/extrap_llm"; os.makedirs(OUTDIR, exist_ok=True)
BSEEDS = list(range(8))
BCORTEX = {"grid": ["--constrained_velocity"], "place": []}
BCOMMON = ["--task", "bearing", "--epochs", "3", "--n_train", "2400", "--n_val", "300",
           "--train_lengths", "6", "8", "10", "12", "--eval_lengths", "8", "16", "24"]
bjobs = [(c, s) for c in BCORTEX for s in BSEEDS]
print(f"bearing: {len(bjobs)} runs (done ones skipped)\n")
for cortex, seed in bjobs:
    out = f"{OUTDIR}/bearing_{cortex}_s{seed}.json"
    if os.path.exists(out):
        print(f"skip  bearing {cortex:5} seed={seed}"); continue
    cmd = (["python", "-u", "-m", "src.training.train_trajectory"] + BCOMMON
           + ["--seed", str(seed), "--out", out] + BCORTEX[cortex])
    print(f"\n>>> bearing {cortex} seed={seed}", flush=True); t0 = time.time()
    subprocess.run(cmd, check=True); print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nbearing sweep pass complete (re-run if any remain)")


# %% [cell 6] BEARING paired significance: grid vs place per seed, per length (bootstrap CI + sign-flip p)
import os, json, math, random
OUTDIR = "results/extrap_llm"; SEEDS = list(range(8)); LENS = [8, 16, 24]


def load(cortex, s):
    f = f"{OUTDIR}/bearing_{cortex}_s{s}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def paired(diffs, iters=20000):
    n = len(diffs); m = sum(diffs) / n
    rng = random.Random(0)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo, hi = boot[int(0.025 * iters)], boot[int(0.975 * iters)]
    # two-sided sign-flip permutation p
    ge = 0
    for _ in range(iters):
        v = sum(d * (1 if rng.random() < 0.5 else -1) for d in diffs) / n
        ge += abs(v) >= abs(m) - 1e-12
    return m, lo, hi, ge / iters


print("BEARING — grid vs place, paired over seeds (cortex-ON exact)\n" + "=" * 64)
out = {}
for T in LENS:
    g = [load("grid", s)["results_by_len"][str(T)]["cortex_on_exact"] for s in SEEDS if load("grid", s)]
    p = [load("place", s)["results_by_len"][str(T)]["cortex_on_exact"] for s in SEEDS if load("place", s)]
    n = min(len(g), len(p))
    if n < 2:
        print(f"  T={T}: need both grid & place for >=2 seeds (have {len(g)},{len(p)})"); continue
    diffs = [g[i] - p[i] for i in range(n)]
    m, lo, hi, pv = paired(diffs)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "ns"
    out[T] = {"grid_mean": round(sum(g[:n]) / n, 4), "place_mean": round(sum(p[:n]) / n, 4),
              "delta": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "p_perm": round(pv, 4), "n": n}
    print(f"  T={T}: grid {sum(g[:n])/n:.0%} vs place {sum(p[:n])/n:.0%} | Δ={m:+.3f} "
          f"[{lo:+.3f},{hi:+.3f}] p={pv:.4f}  {sig}  (n={n})")
json.dump(out, open("results/extrapolation_llm_bearing.json", "w"), indent=2)
print("\nwrote results/extrapolation_llm_bearing.json  <-- paste the table above back")
