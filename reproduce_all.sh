#!/usr/bin/env bash
# reproduce_all.sh — regenerate every CPU result/figure behind the paper from scratch.
#
# Verified environment: Python 3.11, torch 2.2.2 (CPU is fine), numpy (1.26+ or 2.x both work; a
# harmless NumPy-1.x/2.x import warning may print). No GPU needed for any of the MAIN experiments
# below; the LANGUAGE results (Qwen + LoRA) run on a single T4 via notebooks/ (see bottom).
#
#   bash reproduce_all.sh            # MAIN experiments at the paper's seed counts
#   SEEDS=3 bash reproduce_all.sh    # quicker pass (fewer seeds)
#   bash reproduce_all.sh exploratory  # also run the single-run demo pillars
#
# Each command writes results/<name>.json (+ .svg). Artifacts are committed, so a clean run should
# reproduce the committed numbers within seed noise.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SEEDS="${SEEDS:-8}"          # stats/extrapolation paper runs use 8; characterization uses 5
SEEDS5="${SEEDS5:-5}"

run () { echo; echo "=== $* ==="; python -u -m "$@"; }

echo "############ MAIN experiments (multi-seed, paper figures) ############"
run src.eval.stats             --seeds "$SEEDS"     # §6 cognitive suite: planning/value/relational/continual
run src.eval.extrapolation     --seeds "$SEEDS"     # Fig 1: length extrapolation vs fair place/GRU/oracle
run src.eval.ablations         --seeds "$SEEDS5"    # Fig 2a: range/scale/training-dist/sequence-model ablations
run src.eval.seq_baselines     --seeds "$SEEDS5"    # Fig 2b: fair Transformer baselines (the honest tie)
run src.eval.code_necessity    --seeds "$SEEDS5"    # Fig 3: capacity + remapping (where the code wins)
run src.eval.multimap_task     --seeds "$SEEDS5"    # boundary: remapping doesn't help a trained model w/ context-id
run src.eval.frontier_probes   --seeds "$SEEDS5"    # Fig 4: sample efficiency + noise (honest non-wins)
run src.eval.controls          --seeds "$SEEDS5"    # mechanism vs parameters control
run src.eval.significance      --n_fast 20 --n_slow 8   # paired tests (p-values) on every headline claim; resumable
run src.eval.torus             --seeds "$SEEDS"     # non-Euclidean (torus): periodicity NECESSARY, breaks the tie + leakage rebuttal
run src.eval.structural_transfer --seeds "$SEEDS"   # TEM: frozen space-trained metric -> relational inference (+ falsifiers)
python -u -m src.eval.phase_diagram   # SYNTHESIS: when each inductive bias wins (reads the JSONs above; no training)
run src.eval.successor         --seeds "$SEEDS"     # §7 predictive map (SR): plans detours where geometry stalls; geodesic fields; TD-learned
run src.eval.time_cells        --seeds "$SEEDS"     # §7 temporal map: time cells + scalar (Weber) timing EMERGE from a trained recurrent substrate (vs untrained control)
run src.eval.spiking_time_cells --seeds 6           # §7 SPIKING + multi-timescale: spiking time cells + emergent tau-spectrum that aids timing (vs homogeneous-tau control)
run src.eval.eprop_local_learning --seeds 5         # §7 LOCAL LEARNING: time cells emerge under e-prop (eligibility traces, NO backprop) -- the brain's learning rule
run src.eval.space_time_circuit --seeds 5           # §7 CIRCUIT EMBEDDING: place + time + conjunctive space-time cells coexist in ONE circuit (Neuron 2024)
run src.eval.predictions       --seeds 3            # §7 HYPOTHESIS GENERATOR: falsifiable predictions (content->conjunctive; spatial-noise->pure-time)
run src.eval.agent_navigation  --seeds 5            # BEHAVING AGENT: closed-loop navigation emerges + self-learned SR map -> flexible zero-shot any-goal nav
run src.eval.agent_memory      --seeds 5            # BEHAVING AGENT: one-shot place learning (episodic store); lesion abolishes it (Morris water maze)
run src.eval.agent_timing      --seeds 3            # BEHAVING AGENT: interval-timed action (time cells); lesion abolishes timing (reward 0.88->0.00)
run src.eval.content_binding   --seeds 6           # §7 content-binding (what-where-when): conjunctive vs pure time cells + decode what & when (bat CA1 2023)

if [ "${1:-}" = "exploratory" ]; then
  echo; echo "############ EXPLORATORY demos (illustrative; not the central claims) ############"
  for m in emergence boundary_anchoring pillars planning goal_navigation relational continual embodiment generalize_trajectory magnitude_frontier; do
    if python - "$m" <<'PY' 2>/dev/null; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("src.eval." + sys.argv[1]) else 1)
PY
      run "src.eval.$m" || echo "  (skipped src.eval.$m — non-zero exit)"
    fi
  done
fi

echo; echo "############ LANGUAGE results (GPU) ############"
echo "Run on a single T4 (not here): notebooks/m2_extrapolation_multiseed_kaggle.py (multi-seed grid vs place),"
echo "and notebooks/m2_grid_cortex_all_tasks_kaggle.py. See REPRODUCE.md for the figure->command map."
echo; echo "DONE — see results/*.json and results/*.svg"
