# Milestone 2 — harder questions on Kaggle (T4)
# ---------------------------------------------------------------------------
# Beyond the forgiving yes/no "are you back where you started?", we ask the LLM to read
# the actual displacement VECTOR through the (frozen, self-supervised) cortex:
#   - distance : "How far are you from where you started?"  -> bucket 0..5  (MAGNITUDE)
#   - bearing  : "Which direction is the start from here?"  -> compass word (DIRECTION)
# Both use the now-default generalizing recipe (scale-free readout + mixed lengths
# 6,8,10,12) and are evaluated at T = 8, 16, 24 (16,24 are held-out EXTRAPOLATION).
# We report EXACT and WITHIN-1 accuracy (within-1 is circular for bearing), cortex ON
# vs OFF (text-only control), per length. CPU pre-check expects bearing to generalize
# cleanly (scale-invariant) and distance to be solid but degrade with length (its scale
# must extrapolate). Copy each block into its own Kaggle cell.
# ===========================================================================


# %% [cell 1] setup — env, repo, deps
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # pin ONE gpu (avoid the T4x2 DataParallel hang)
os.environ["HF_HUB_DISABLE_XET"] = "1"       # avoid the Xet download stall
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
!pip -q uninstall -y torchao   # peft's torchao>=0.16 guard trips on Kaggle's old 0.10
print("setup done")


# %% [cell 2] pre-download the base LLM (so training doesn't stall mid-run)
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] DISTANCE — "how far from the start?" (quantized bucket 0..5)
# Watch the probe line first: it previews whether magnitude is decodable from the rep.
!python -u -m src.training.train_trajectory --task distance --n_train 2400 --n_val 300 --epochs 3 --out results/m2_distance.json


# %% [cell 4] BEARING — "which way is home?" (8-way compass; scale-invariant)
!python -u -m src.training.train_trajectory --task bearing --n_train 2400 --n_val 300 --epochs 3 --out results/m2_bearing.json


# %% [cell 5] show both results (paste this output back to commit it to main)
import json
for f in ["results/m2_distance.json", "results/m2_bearing.json"]:
    print("=" * 72); print(f, "  task =", json.load(open(f))["task"])
    d = json.load(open(f))
    print("  probe acc by length :", d["probe_acc_by_len"])
    for T, r in d["results_by_len"].items():
        tag = "EXTRAPOLATION" if r["extrapolation"] else "train-range "
        print(f"  T={T:>2} [{tag}]  exact ON={r['cortex_on_exact']:.1%} OFF={r['cortex_off_exact']:.1%}"
              f"   within1 ON={r['cortex_on_within1']:.1%} OFF={r['cortex_off_within1']:.1%}"
              f"   (chance~{r['chance']:.0%})")


# %% [cell 6] CAPSTONE — the magnitude FIX, biologically faithful: distance with a
# self-supervised GRID-CELL cortex (NO coordinate labels). The CPU sweep
# (src/eval/magnitude_frontier.py) found bounded PLACE cells are the magnitude bottleneck,
# while the periodic multi-scale GRID code extrapolates magnitude (cortex probe 93/83/64% vs
# place 91/72/45%). This confirms the faithful fix on the full LLM. ~1.5 h on a T4.
!python -u -m src.training.train_trajectory --task distance --code grid --n_train 2400 --n_val 300 --epochs 3 --out results/m2_distance_grid.json


# %% [cell 7] compare PLACE (cell 3) vs GRID (cell 6) for distance — paste this back
import json, os
print("distance: PLACE-cell vs GRID-cell self-supervised cortex (exact / within-1, by length)")
for label, f in [("place", "results/m2_distance.json"), ("grid ", "results/m2_distance_grid.json")]:
    if not os.path.exists(f):
        print(f"  {label}: (missing {f})"); continue
    d = json.load(open(f))
    print(f"  {label}  probe={d['probe_acc_by_len']}")
    for T, r in d["results_by_len"].items():
        tag = "EXTRAP" if r["extrapolation"] else "train "
        print(f"    T={T:>2} [{tag}]  exact ON={r['cortex_on_exact']:.0%} OFF={r['cortex_off_exact']:.0%}"
              f"   within1 ON={r['cortex_on_within1']:.0%}   (chance~{r['chance']:.0%})")
