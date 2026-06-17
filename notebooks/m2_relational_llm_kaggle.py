# Milestone 2 — STRUCTURAL TRANSFER through a frozen LLM (the TEM headline, Kaggle T4)
# ---------------------------------------------------------------------------
# A space-only-trained cortex is FROZEN. An abstract ordered structure (ranks) is laid along a concept
# axis; each item enters Qwen by its OWN position through the frozen cortex (never the signed relative
# displacement -> no leak). A LoRA-Qwen reads BOTH items' spatial tokens and answers a LINGUISTIC
# comparison, trained on ADJACENT pairs only. Test: transitive inference on never-seen FAR pairs, plus
# falsifiers (shuffled positions, scrambled 2nd item) and the cortex-OFF text-only control.
#
# Design validated on CPU first (src/eval/structural_transfer_cortex.py): through the actual frozen
# cortex.encode, TI = 0.99 and controls collapse. NOTE: the two-item LLM trainer is new — if a run errors,
# it likely needs one quick debug pass (it reuses the proven TrajectoryLLM cortex/fusion/LoRA path).
# RESUMABLE: skips seeds already written. Run alongside the torus cell in the same session.
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


# %% [cell 3] structural-transfer sweep (resumable). seeds 0..2; ~20-40 min each on T4.
import os, subprocess, time
OUTDIR = "results/relational_llm"; os.makedirs(OUTDIR, exist_ok=True)
for seed in [0, 1, 2]:
    out = f"{OUTDIR}/relational_s{seed}.json"
    if os.path.exists(out):
        print(f"skip seed={seed}"); continue
    cmd = ["python", "-u", "-m", "src.training.train_relational",
           "--n_items", "12", "--epochs", "4", "--seed", str(seed), "--out", out]
    print(f"\n>>> relational seed={seed}", flush=True); t0 = time.time()
    subprocess.run(cmd, check=True); print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nrelational sweep pass complete (re-run if any remain)")


# %% [cell 4] aggregate -> mean +/- 95% CI + paired TI-vs-shuffled test (paste this back)
import os, json, math, random
OUTDIR = "results/relational_llm"; SEEDS = list(range(3))
KEYS = ["transitive_inference_far", "adjacent_trained", "far_shuffled_positions",
        "far_scrambled_2nd", "far_cortex_OFF"]


def ci95(xs):
    n = len(xs); m = sum(xs)/n
    sd = (sum((x-m)**2 for x in xs)/(n-1))**0.5 if n > 1 else 0.0
    return m, 1.96*sd/math.sqrt(n)


rows = [json.load(open(f"{OUTDIR}/relational_s{s}.json"))["results"] for s in SEEDS
        if os.path.exists(f"{OUTDIR}/relational_s{s}.json")]
print(f"STRUCTURAL TRANSFER through the frozen LLM (n={len(rows)} seeds; chance=50%)")
print("=" * 64)
for k in KEYS:
    xs = [r[k] for r in rows if k in r]
    if xs:
        m, c = ci95(xs); print(f"  {k:28} {m:.1%} +/- {c:.1%}")
# paired TI vs shuffled-positions
if rows:
    d = [r["transitive_inference_far"] - r["far_shuffled_positions"] for r in rows]
    rng = random.Random(0); n = len(d); m = sum(d)/n
    p = sum(abs(sum(x*(1 if rng.random()<0.5 else -1) for x in d)/n) >= abs(m)-1e-12 for _ in range(20000))/20000
    print(f"\n  TI(ordered) - TI(shuffled) = {m:+.1%}  paired p={p:.3f}")
json.dump({"seeds": SEEDS, "results_by_seed": rows}, open("results/relational_llm.json", "w"), indent=2)
print("\nwrote results/relational_llm.json  <-- paste the table above back")
print("Headline: TI(far) >> cortex-OFF (~50%) and >> shuffled-positions = a SPACE-trained frozen code,")
print("read by a frozen LLM, does ABSTRACT transitive inference it cannot do text-only (the TEM claim).")
