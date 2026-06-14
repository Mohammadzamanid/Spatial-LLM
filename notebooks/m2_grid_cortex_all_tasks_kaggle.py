# Milestone 2 — the faithful GRID-CELL cortex on ALL THREE language tasks (Kaggle T4)
# ---------------------------------------------------------------------------
# Run return + bearing + distance through the velocity-driven hexagonal grid-cell cortex
# (--constrained_velocity), and compare each task to its place/default-cortex baseline.
# Goal: show the biologically-faithful cortex (emergent hexagonal grids, length-invariant,
# metrically accurate) carries the LLM across every navigation question — the M2 cortex,
# consolidated. Copy each block into its own Kaggle cell. ~1.5 h per training run on a T4.
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


# %% [cell 3] RETURN — "are you back where you started?" through the grid-cell cortex
!python -u -m src.training.train_trajectory --task return --constrained_velocity --n_train 2400 --n_val 300 --epochs 3 --out results/m2_return_gridcortex.json


# %% [cell 4] BEARING — "which way is home?" (8-way compass) through the grid-cell cortex
!python -u -m src.training.train_trajectory --task bearing --constrained_velocity --n_train 2400 --n_val 300 --epochs 3 --out results/m2_bearing_gridcortex.json


# %% [cell 5] DISTANCE — "how far?" through the grid-cell cortex (re-run for a self-contained sweep)
!python -u -m src.training.train_trajectory --task distance --constrained_velocity --n_train 2400 --n_val 300 --epochs 3 --out results/m2_distance_gridcortex.json


# %% [cell 5b] DISTANCE RE-RUN — the 6-class task is prone to collapse-to-prior (ON==OFF). Give the
# LLM more epochs (and a different seed) to open onto the cortex. The cortex rep is fine (probe
# 98/96/91); this is an optimization fix. Run this if cell 5 gave ON==OFF ~chance.
!python -u -m src.training.train_trajectory --task distance --constrained_velocity --n_train 2400 --n_val 300 --epochs 6 --seed 0 --out results/m2_distance_gridcortex.json


# %% [cell 6] compare grid-cell cortex vs place/default cortex on all three tasks (paste this back)
import json, os
pairs = [("return",   "results/m2_return_gridcortex.json",   "results/m2_lengthgen_scalefree_mixed.json"),
         ("bearing",  "results/m2_bearing_gridcortex.json",  "results/m2_bearing.json"),
         ("distance", "results/m2_distance_gridcortex.json", "results/m2_distance.json")]
for task, gridf, basef in pairs:
    print("=" * 72); print(f"TASK = {task}")
    for label, f in [("GRID-CELL cortex", gridf), ("place/default  ", basef)]:
        if not os.path.exists(f):
            print(f"  {label}: (missing {f})"); continue
        d = json.load(open(f)); r = d["results_by_len"]
        def fmt(T):
            x = r.get(str(T)) or r.get(T)
            if not x: return f"T{T}:--"
            on = x.get("cortex_on_exact", x.get("cortex_on"))
            off = x.get("cortex_off_exact", x.get("cortex_off"))
            return f"T{T}:{on:.0%}/{off:.0%}"
        print(f"  {label}  " + "  ".join(fmt(T) for T in (8, 16, 24)) + "   (ON/OFF exact)")
