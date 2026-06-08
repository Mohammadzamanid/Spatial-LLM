#!/usr/bin/env bash
# Multi-seed train+eval sweep for ONE config. Foreground by design — launch it in
# the background so it survives a closed browser/SSH session:
#
#   nohup bash scripts/run_sweep.sh configs/coord_2d_noleak.yaml coord2d 42 43 44 \
#       > sweep_coord2d.log 2>&1 &
#   tail -f sweep_coord2d.log        # watch; Ctrl-C stops watching, NOT the sweep
#
# Each run writes results/<label>_seed<seed>.json (paste-back ready). Collect with:
#   python scripts/collect_results.py coord2d
#
# Portable across Kaggle / Lightning AI / SageMaker Studio Lab / Paperspace / any VM.
set -euo pipefail

export HF_HUB_DISABLE_XET=1                       # avoid the Xet streaming stall
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single GPU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [ "$#" -lt 2 ]; then
  echo "usage: bash scripts/run_sweep.sh <config.yaml> <label> [seed ...]" >&2
  exit 1
fi
CFG="$1"; LABEL="$2"; shift 2
SEEDS=("$@"); [ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(42 43 44)

mkdir -p results outputs
for s in "${SEEDS[@]}"; do
  OUT="outputs/${LABEL}_seed${s}"
  echo "===== TRAIN ${LABEL} seed=${s} -> ${OUT} ====="
  python -u -m src.training.trainer --config "$CFG" --seed "$s" --output_dir "$OUT"
  echo "===== EVAL  ${LABEL} seed=${s} ====="
  python -m src.eval.accuracy --config "$CFG" --checkpoint "$OUT" \
    --val data/processed/val.jsonl --dump-gates --seed "$s" \
    --label "${LABEL}_seed${s}" --results-json "results/${LABEL}_seed${s}.json"
done
echo "===== SWEEP DONE: results/${LABEL}_seed*.json ====="
