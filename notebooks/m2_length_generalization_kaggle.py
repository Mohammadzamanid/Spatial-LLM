# Milestone 2 — length generalization on Kaggle (T4)
# ---------------------------------------------------------------------------
# Head-to-head: does TrajectoryLLM answer correctly about paths LONGER than it
# trained on?  We compare the old M2 recipe (fixed length + readout(u/T)) against
# the fix from the generalization stress-test (mixed lengths + scale-free readout).
# Both train on SHORT paths and are evaluated at T = 8, 16, 24 — 16 and 24 are
# held-out EXTRAPOLATION (longer than anything trained on in the fix; 2-3x the
# single training length in the baseline).
#
# Copy each block below into its own Kaggle cell. GPU: T4 x2 (we pin to one).
# ===========================================================================


# %% [cell 1] setup — env, repo, deps
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # pin ONE gpu (avoid the T4x2 DataParallel hang)
os.environ["HF_HUB_DISABLE_XET"] = "1"       # avoid the Xet download stall
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
print("setup done")


# %% [cell 2] pre-download the base LLM (so training doesn't stall mid-run)
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] RUN A — baseline: fixed length (T=8) + readout(u/T)  [the old M2 recipe]
# Expectation: strong at T=8, degrades at the longer held-out lengths (length-locked).
!python -u -m src.training.train_trajectory --train_lengths 8 --eval_lengths 8 16 24 --cortex_pretrain selfsup --n_train 2400 --n_val 300 --epochs 3 --out results/m2_lengthgen_baseline.json


# %% [cell 4] RUN B — fix: mixed lengths (6,8,10,12) + scale-free readout(u)
# Expectation: accuracy HOLDS at the held-out longer lengths (16, 24) — the cortex
# learned the length-invariant integration operation, not one length.
!python -u -m src.training.train_trajectory --train_lengths 6 8 10 12 --eval_lengths 8 16 24 --cortex_scale_free --cortex_pretrain selfsup --n_train 2400 --n_val 300 --epochs 3 --out results/m2_lengthgen_scalefree_mixed.json


# %% [cell 5] show both results (paste this output back to commit it to main)
import json
for f in ["results/m2_lengthgen_baseline.json", "results/m2_lengthgen_scalefree_mixed.json"]:
    print("=" * 70); print(f)
    d = json.load(open(f))
    print("  probe acc by length :", d["probe_acc_by_len"])
    for T, r in d["results_by_len"].items():
        tag = "EXTRAPOLATION" if r["extrapolation"] else "train-range "
        print(f"  T={T:>2} [{tag}]  cortex ON={r['cortex_on']:.1%}  OFF={r['cortex_off']:.1%}")
