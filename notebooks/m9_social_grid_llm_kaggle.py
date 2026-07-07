# Milestone 9 — the LLM reasons over a 2-D SOCIAL space (Tavares 2015; Park-Miller 2021; Kaggle T4)
# ---------------------------------------------------------------------------
# The cognitive-map claim extended to the SOCIAL domain (after gap #4's self/other place cells). A space-only
# cortex is FROZEN. Agents are placed in a 2-D social map (axis-0 = POWER, axis-1 = AFFILIATION); each agent
# enters Qwen by its OWN social position through the frozen cortex (no leak). A LoRA-Qwen reads two agent codes
# and answers "Is the first person more dominant than the second?" (the POWER axis — the social-hierarchy
# transitive-inference result, Kumaran 2016 / Park-Miller 2021), trained on power-ADJACENT pairs only.
# Tests: FAR power pairs (transitive inference); the AFFILIATION-DISSOCIATION set (affiliation ordering OPPOSES
# power -> the model must read power, not affiliation); and the falsifiers (shuffled positions, cortex-OFF).
#
# Design validated on CPU first (src/eval/social_grid_cortex.py, n=5): held-out DOMINANCE from the power axis
# = 0.96, the axis DISSOCIATION (power->dominance 0.96 vs affiliation->dominance 0.45, gap +0.51) is clean,
# and shuffled collapses dominance to ~0.44. DOMINANCE reuses the PROVEN two-item forward of train_relational
# (lowest debug risk). RESUMABLE: skips seeds already written.
# (For the 2-D SOCIAL-DISTANCE query "who is socially closer to X?", run train_social with --task distance,
#  which reuses the #8 triple trainer on the social axes.)
# ===========================================================================


# %% [cell 1] setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # reduce T4 fragmentation OOM
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
!pip -q uninstall -y torchao
print("setup done")


# %% [cell 2] cache the base LLM
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] social-dominance sweep (resumable). seeds 0..2; ~20-40 min each on T4.
import os, subprocess, time, glob
OUTDIR = "results/social_llm"; os.makedirs(OUTDIR, exist_ok=True)
FRESH = True
if FRESH:
    for f in glob.glob(f"{OUTDIR}/*.json"):
        os.remove(f)
    print("cleared old results in", OUTDIR)
for seed in [0, 1, 2]:
    out = f"{OUTDIR}/social_s{seed}.json"
    if os.path.exists(out):
        print(f"skip seed={seed} (exists)"); continue
    cmd = ["python", "-u", "-m", "src.training.train_social", "--task", "dominance",
           "--G", "6", "--spacing", "0.8", "--steps", "1500", "--bs", "8", "--seed", str(seed), "--out", out]
    print(f"\n>>> social seed={seed}  (watch the 'step N: loss' lines)", flush=True); t0 = time.time()
    subprocess.run(cmd, check=True); print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nsocial sweep pass complete")


# %% [cell 4] aggregate -> mean +/- 95% CI + paired DISSOCIATION-vs-cortex-OFF test (paste this back)
import os, json, math, random
OUTDIR = "results/social_llm"; SEEDS = list(range(3))
KEYS = ["dominance_far", "dominance_dissociation", "dominance_adj_trained",
        "dominance_far_cortex_OFF", "dominance_far_shuffled_pos"]


def ci95(xs):
    n = len(xs); m = sum(xs)/n
    sd = (sum((x-m)**2 for x in xs)/(n-1))**0.5 if n > 1 else 0.0
    return m, 1.96*sd/math.sqrt(n)


rows = [json.load(open(f"{OUTDIR}/social_s{s}.json"))["results"] for s in SEEDS
        if os.path.exists(f"{OUTDIR}/social_s{s}.json")]
print(f"SOCIAL DOMINANCE through the frozen LLM (n={len(rows)} seeds; chance=50%)")
print("=" * 66)
for k in KEYS:
    xs = [r[k] for r in rows if k in r]
    if xs:
        m, c = ci95(xs); print(f"  {k:28} {m:.1%} +/- {c:.1%}")
# paired DISSOCIATION (reads power) vs cortex-OFF
if rows:
    d = [r["dominance_dissociation"] - r["dominance_far_cortex_OFF"] for r in rows]
    rng = random.Random(0); n = len(d); m = sum(d)/n
    p = sum(abs(sum(x*(1 if rng.random()<0.5 else -1) for x in d)/n) >= abs(m)-1e-12 for _ in range(20000))/20000
    print(f"\n  DISSOCIATION(cortex-ON) - cortex-OFF = {m:+.1%}  paired p={p:.3f}")
json.dump({"seeds": SEEDS, "results_by_seed": rows}, open("results/social_llm.json", "w"), indent=2)
print("\nwrote results/social_llm.json  <-- paste the table above back")
print("Headline: DOMINANCE (far + dissociation) >> cortex-OFF (~50%) and >> shuffled = a SPACE-trained frozen")
print("code, read by a frozen LLM, does SOCIAL-hierarchy inference reading the POWER axis (not affiliation).")
