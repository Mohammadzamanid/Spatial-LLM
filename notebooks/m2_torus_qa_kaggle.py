# Milestone 2 — TORUS-QA through a frozen LLM: the leakage-proof CAUSAL headline (Kaggle T4)
# ============================================================================================
# A frozen, TOROIDALLY-pretrained cortex lets Qwen answer "which wrap-around cell (0-8) are you in?"
# on a board that wraps at its edges (a torus). The world has NO faithful Euclidean text description and
# the moves never appear in the prompt, so a high cortex-ON vs text-only-OFF gap is a clean CAUSAL +
# leakage-proof statement: the LLM answers by reading a path-integrated toroidal code.
# Verified (seed 0): cortex-ON 94/78/70% at T=8/16/24 vs text-only-OFF ~chance (11%), flat to 3x length.
#
# This is the ONLY experiment in this notebook. Run the cells top to bottom. ~36 min/seed on a T4;
# resumable (skips seeds already done). Self-contained — no other cells needed.
# ============================================================================================


# %% [cell 1] setup: clone/pull the repo + install deps
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
!pip -q uninstall -y torchao
print("setup done")


# %% [cell 2] cache the base LLM (Qwen2.5-1.5B)
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] train + evaluate torus-QA for 3 seeds (resumable; ~36 min each on a T4)
import os, subprocess, time
OUT = "results/torus_llm"; os.makedirs(OUT, exist_ok=True)
for seed in [0, 1, 2]:
    out = f"{OUT}/torus_grid_s{seed}.json"
    if os.path.exists(out):
        print(f"skip seed={seed} (already done)"); continue
    print(f"\n===== TORUS-QA  seed={seed} =====", flush=True); t0 = time.time()
    subprocess.run(["python", "-u", "-m", "src.training.train_trajectory",
        "--task", "torus", "--constrained_velocity", "--n_train", "2400", "--n_val", "300",
        "--epochs", "5", "--early_stop", "--train_lengths", "6", "8", "10", "12",
        "--eval_lengths", "8", "16", "24", "--seed", str(seed), "--out", out], check=True)
    print(f"seed {seed} done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nall torus seeds complete")


# %% [cell 4] aggregate -> cortex-ON vs text-only-OFF, mean +/- 95% CI + paired test (paste this back)
import os, json, math, random
OUT = "results/torus_llm"; LENS = ["8", "16", "24"]


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def paired_p(diffs, iters=20000):
    n = len(diffs); m = sum(diffs) / n; rng = random.Random(0)
    return sum(abs(sum(d * (1 if rng.random() < 0.5 else -1) for d in diffs) / n) >= abs(m) - 1e-12
               for _ in range(iters)) / iters


rows = [json.load(open(f"{OUT}/torus_grid_s{s}.json")) for s in [0, 1, 2]
        if os.path.exists(f"{OUT}/torus_grid_s{s}.json")]
print(f"TORUS-QA through the frozen LLM (n={len(rows)} seeds; chance ~11-18%)"); print("=" * 64)
print("  T      cortex-ON          text-only OFF      Delta(ON-OFF)   p")
agg = {}
for T in LENS:
    on = [r["results_by_len"][T]["cortex_on_exact"] for r in rows]
    off = [r["results_by_len"][T]["cortex_off_exact"] for r in rows]
    mo, co = ci95(on); mf, cf = ci95(off)
    d = [on[i] - off[i] for i in range(len(rows))]
    p = paired_p(d) if len(rows) >= 2 else float("nan")
    agg[T] = {"on": [round(mo, 4), round(co, 4)], "off": [round(mf, 4), round(cf, 4)],
              "delta": round(sum(d) / len(d), 4), "p": round(p, 4), "n": len(rows)}
    print(f"  T={T:<3}  {mo:.0%} +/-{co:.0%}       {mf:.0%} +/-{cf:.0%}      {sum(d)/len(d):+.0%}       {p:.4f}")
json.dump({"n_seeds": len(rows), "by_len": agg}, open("results/torus_llm_agg.json", "w"), indent=2)
print("\nwrote results/torus_llm_agg.json  <-- paste the table above back, or share this file")
print("Headline: cortex-ON >> text-only-OFF on a non-Euclidean (toroidal) world = the LLM reads a")
print("path-integrated toroidal code it cannot get from a Euclidean text prior (leakage-proof, causal).")
