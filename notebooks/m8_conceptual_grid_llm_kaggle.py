# Milestone 8 — the LLM reads a 2-D CONCEPTUAL grid (Constantinescu/Behrens 2016; Kaggle T4)
# ---------------------------------------------------------------------------
# The cognitive-map claim extended from SPACE to MEANING at the language level. A space-only-trained cortex
# is FROZEN. Concepts are laid at 2-D coordinates; each concept enters Qwen by its OWN position through the
# frozen cortex (heading=atan2(y,x), speed=r/T -> no leak). A LoRA-Qwen reads THREE concept codes (anchor +
# two candidates) and answers the LINGUISTIC comparison "Is the FIRST closer to the ANCHOR than the SECOND?",
# trained on NEAR triples only. Test: 2-D "closer" reasoning on never-seen FAR / OFF-AXIS triples (where a
# 1-D projection code is <=50% by construction), plus the falsifiers (shuffled positions, cortex-OFF text-only).
#
# Design validated on CPU first (src/eval/conceptual_grid_cortex.py, n=5): on the ACTUAL frozen cortex.encode
# the LLM reads, OFF-AXIS "closer" = 0.65 (>chance), held-out decode 0.63 vs shuffled 3.4 spacing, shuffled
# Spearman ~0 — a control-clean 2-D metric. The trained LLM readout is expected to SHARPEN this (1-D precedent
# structural_transfer -> train_relational: 1.0 -> 0.99). NOTE: the THREE-item forward is new vs the proven
# two-item train_relational — if a run errors it likely needs one quick debug pass (it reuses the same
# TrajectoryLLM cortex / to_tokens / gated-fusion / LoRA path). RESUMABLE: skips seeds already written.
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


# %% [cell 3] conceptual-grid sweep (resumable). seeds 0..2; ~25-45 min each on T4.
import os, subprocess, time, glob
OUTDIR = "results/conceptual_llm"; os.makedirs(OUTDIR, exist_ok=True)
FRESH = True   # True = clear old JSONs and run all seeds fresh (prevents reading stale/degenerate results)
if FRESH:
    for f in glob.glob(f"{OUTDIR}/*.json"):
        os.remove(f)
    print("cleared old results in", OUTDIR)
for seed in [0, 1, 2]:
    out = f"{OUTDIR}/conceptual_s{seed}.json"
    if os.path.exists(out):
        print(f"skip seed={seed} (exists)"); continue
    cmd = ["python", "-u", "-m", "src.training.train_conceptual",
           "--G", "6", "--spacing", "0.8", "--steps", "4000", "--bs", "6", "--seed", str(seed), "--out", out]
    print(f"\n>>> conceptual seed={seed}  (watch the 'step N: loss' lines)", flush=True); t0 = time.time()
    subprocess.run(cmd, check=True); print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nconceptual sweep pass complete")


# %% [cell 3b] RE-EVAL an existing checkpoint with the FIXED eval (padding-immune, balanced, capped) — FREE,
# no retraining. Use this to salvage a prior run: if closer_near_trained >> 50% the model LEARNED and the old
# candidate-NLL eval was the bug; if it is ~50% the readout genuinely underfit (raise --bs/--steps). Minutes.
import glob, subprocess
for pt in sorted(glob.glob("results/conceptual_llm/conceptual_s*.pt")):
    seed = pt.split("conceptual_s")[1].split(".pt")[0]
    print(f"\n>>> re-eval {pt} (seed {seed})", flush=True)
    subprocess.run(["python", "-u", "-m", "src.training.train_conceptual", "--reeval", pt,
                    "--G", "6", "--spacing", "0.8", "--seed", seed], check=True)


# %% [cell 4] aggregate -> mean +/- 95% CI + paired OFF-AXIS-vs-cortex-OFF test (paste this back)
import os, json, math, random
OUTDIR = "results/conceptual_llm"; SEEDS = list(range(3))
KEYS = ["closer_far", "closer_far_OFFAXIS", "closer_near_trained",
        "closer_far_cortex_OFF", "closer_far_shuffled_pos"]


def ci95(xs):
    n = len(xs); m = sum(xs)/n
    sd = (sum((x-m)**2 for x in xs)/(n-1))**0.5 if n > 1 else 0.0
    return m, 1.96*sd/math.sqrt(n)


rows = [json.load(open(f"{OUTDIR}/conceptual_s{s}.json"))["results"] for s in SEEDS
        if os.path.exists(f"{OUTDIR}/conceptual_s{s}.json")]
print(f"CONCEPTUAL GRID through the frozen LLM (n={len(rows)} seeds; chance=50%)")
print("=" * 66)
for k in KEYS:
    xs = [r[k] for r in rows if k in r]
    if xs:
        m, c = ci95(xs); print(f"  {k:26} {m:.1%} +/- {c:.1%}")
# paired OFF-AXIS vs cortex-OFF
if rows:
    d = [r["closer_far_OFFAXIS"] - r["closer_far_cortex_OFF"] for r in rows]
    rng = random.Random(0); n = len(d); m = sum(d)/n
    p = sum(abs(sum(x*(1 if rng.random()<0.5 else -1) for x in d)/n) >= abs(m)-1e-12 for _ in range(20000))/20000
    print(f"\n  OFF-AXIS(cortex-ON) - cortex-OFF = {m:+.1%}  paired p={p:.3f}")
json.dump({"seeds": SEEDS, "results_by_seed": rows}, open("results/conceptual_llm.json", "w"), indent=2)
print("\nwrote results/conceptual_llm.json  <-- paste the table above back")
print("Headline: OFF-AXIS closer >> cortex-OFF (~50%) and >> shuffled = a SPACE-trained frozen code, read by")
print("a frozen LLM, does 2-D CONCEPTUAL reasoning ('closer') it cannot do text-only (the map, from space to meaning).")
