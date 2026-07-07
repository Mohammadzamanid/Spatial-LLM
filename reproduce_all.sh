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
run src.eval.btsp              --seeds 5            # §7 BTSP one-shot learning (Bittner 2017): one plateau imprints a place field in ONE pass via a seconds-wide asymmetric kernel; field shifts UPSTREAM (predictive), shift scales with speed; needs seconds-scale + asymmetry (vs STDP/symmetric controls). GAPS.md #1
run src.eval.space_time_circuit --seeds 5           # §7 CIRCUIT EMBEDDING: place + time + conjunctive space-time cells coexist in ONE circuit (Neuron 2024)
run src.eval.social_space      --seeds 5            # SOCIAL SPACE (Danjo 2018, Omer 2018): one substrate fed self + OTHER-agent motion develops pure self-place + pure other-place cells (emergent, eta^2); self/other lesion double dissociation. GAPS.md #4
run src.eval.goal_vector       --seeds 5            # GOAL-VECTOR CODE (Sarel 2017; Banino 2018): a policy trained ONLY to reach randomized goals develops a goal-DIRECTION code (emergent, goal-specific vs untrained + shuffle nulls); honest scope (allocentric/redundant; ego+distance don't emerge). GAPS.md #3A
run src.eval.reward_map        --seeds 5            # REWARD MAP (Hollup 2001; Gauthier-Tank 2018): reward-triggered BTSP builds place fields that ANTICIPATE the goal (upstream shift, vanishes with symmetric kernel) + reward-specific over-representation (vs yoked-random control). GAPS.md #3B
run src.eval.hexadirectional   --seeds 5            # HEXADIRECTIONAL / GRID CODE FOR CONCEPTS (Doeller 2010; Constantinescu 2016): hex grid -> 6-fold direction signal via a movement nonlinearity (conjunctive grid x direction); symmetry INHERITED from the lattice (square -> 4-fold), linear read-out flat -> not circular. GAPS.md #2
run src.eval.neuromodulation   --seeds 5            # NEUROMODULATION (Hasselmo 2006; Yu-Dayan 2005): ACh sets encode/retrieve on a CA3 auto-associator -> high-ACh encoding blocks OVERLAP-SPECIFIC intrusion of an old memory at MATCHED write energy (recurrent contamination, not non-storage), completion needs W_rec; NE surprise = NOVELTY not change (AUC~1) and a surprise remap is adaptive two-sided (learns new env + protects old) vs a matched no-reset control. GAPS.md #5
run src.eval.credit_assignment --seeds 5            # CREDIT ASSIGNMENT w/o BACKPROP (Lillicrap 2016; Sacramento 2018; Payeur 2021): a deep spatial module trained by FEEDBACK ALIGNMENT (fixed random backward path, no weight transport) reaches backprop's decode + representation; the forward weights ALIGN to the feedback (grad-align >0 vs shuffled ~0); shuffling the feedback each step collapses learning (falsifier). GAPS.md Tier 5 #A1
run src.eval.meta_learning     --seeds 5            # META-LEARNING / SELF-TUNED LEARNING RATE (Behrens 2007; Wang 2018): a GRU meta-trained ONLY to predict the next obs develops, in frozen recurrent dynamics, a learning rate that RISES with volatility and FALLS with stochasticity (the dissociation, despite highest variance) -- untrained-flat control, beats best fixed-alpha. GAPS.md Tier 5 #B3. (slow: trains 5 nets)
run src.eval.astrocyte_plasticity --seeds 8         # ASTROCYTE-GATED SLOW PLASTICITY (Williamson 2024): a slow glial gate on e-prop that throttles importance-tagged synapses cuts forgetting on a continual stream, beating a UNIFORM reduction at MATCHED plasticity (targeting, not less learning) -- needs the SLOW timescale (fast-astro falsifier ~0); honest recency trade-off; kin to EWC/SI. GAPS.md Tier 5 #B4
run src.eval.emergent_grid_bio --seeds 5            # FAITHFULNESS CAPSTONE (Murray 2019 RFLO + A1 feedback alignment): the REAL path-integration grid cortex trained by a LOCAL no-weight-transport rule (eligibility x fixed random feedback, no BPTT) learns path integration AND grows the emergent periodic grid code (~backprop), never in the loss; shuffled-feedback falsifier falls to the untrained floor. The core itself learns biologically. GAPS.md Tier 5 capstone
run src.eval.complex_synapse   --seeds 5            # MULTI-TIMESCALE SYNAPSE (Benna-Fusi 2016): a synapse built as a CHAIN of coupled variables at geometric timescales forgets as a POWER LAW (log-log straight, slope ~-0.5 = 1/sqrt(t)) where a leaky SCALAR forgets EXPONENTIALLY (semilog straight); 3.3x longer memory at matched initial SNR; lifetime grows with chain depth. Graceful forgetting from the synapse. GAPS.md Tier 5 #B2
run src.eval.predictions       --seeds 3            # §7 HYPOTHESIS GENERATOR: falsifiable predictions (content->conjunctive; spatial-noise->pure-time)
run src.eval.agent_navigation  --seeds 5            # BEHAVING AGENT: closed-loop navigation emerges + self-learned SR map -> flexible zero-shot any-goal nav
run src.eval.agent_memory      --seeds 5            # BEHAVING AGENT: one-shot place learning (episodic store); lesion abolishes it (Morris water maze)
run src.eval.agent_timing      --seeds 3            # BEHAVING AGENT: interval-timed action (time cells); lesion abolishes timing (reward 0.88->0.00)
run src.eval.agent_unified     --seeds 3            # BEHAVING AGENT: ONE task needs space+memory+time; clean triple-lesion dissociation (99% -> 0/0/0)
run src.eval.agent_grid_cortex --seeds 3            # BEHAVING AGENT on the REAL grid cortex: velocity-driven hex grid modules path-integrate -> nonlinear readout -> vector nav; triple dissociation on the grid substrate
run src.eval.agent_grid_drift  --seeds 3            # GRID DRIFT + CORRECTION: noisy self-motion drifts the grid estimate (unbounded); boundary-vector cells reset it near walls (Hardcastle 2015) -> localization bounded + foraging rescued
run src.eval.agent_cue_integration --seeds 3        # OPTIMAL CUE INTEGRATION: a learned fuser (no hand-coded gate) discovers near-optimal Bayesian combination of grid-PI + boundary cues -> beats either cue alone, sits on the optimal bound (Nardini 2008)
run src.eval.head_direction    --seeds 5            # HEAD-DIRECTION ORGAN: HD cells + a ring attractor EMERGE from angular path integration; heading-dominated drift driven by the HD system, bounded by a visual reset (Knierim 1995)
run src.eval.agent_deadreckoning --seeds 3          # DEAD-RECKONING BRAIN: unified HD->grid->place stack from self-motion alone; heading drift propagates to position drift; both allothetic corrections needed; homing abolished by lesioning HD or grid
run src.eval.basal_ganglia     --seeds 3            # TIER-2 SYSTEM: basal-ganglia Go/NoGo action selection; dopamine lesion abolishes learning (100->35%)
run src.eval.grid_capacity     --seeds 5            # WHY GRID CELLS: coding capacity (Fiete) via Fisher info -- grid resolution flat vs arena, place linear (33x at scale); + honest linear-decode caveat
run src.eval.grid_catastrophe   --seeds 5            # CATASTROPHIC ERRORS (Fiete other half): multi-module grid code suppresses catastrophic decode jumps exponentially with module count; bimodal error law; grid dominates place at matched budget
run src.eval.reference_frame   --seeds 5            # MULTI-REFERENCE-FRAME MAP: egocentric object-vector cells (Hoydal 2019) + grid REANCHORING to an object frame; object-relative goal (moving object) solved by object-frame agent, not global; grid translates with the object
run src.eval.plane_of_motion  --seeds 5            # 3D PLANE-OF-MOTION (bat 2026): 2D grid aligned to the PCA-estimated motion plane + off-plane code; orientation-invariant 3D localization; fixed-plane grid fails at steep tilt (honest: no clean win vs naive 3D grid)
run src.eval.landmark_anchoring --seeds 3        # DYNAMIC REFERENCE-FRAME ANCHORING: reliability-gated reanchoring of the grid phase to a landmark (anchor - R(heading)@ego); corrects allocentric drift; allocentric + egocentric codes coexist (MEC 2025)
run src.eval.agent_multiframe  --seeds 3         # UNIFIED MULTI-REFERENCE-FRAME AGENT: ONE brain navigates GLOBAL (grid) AND OBJECT (object-vector+HD) frames; clean double dissociation (-grid kills global, -object kills object, -HD kills both)
run src.eval.theta_sweep     --seeds 5            # THETA-CYCLE LOOK-AROUND (Vollan 2025): online grid sweeps as active look-ahead (alternating L/R, ~20%-spacing, multi-scale); sampling space AHEAD avoids dead-ends a reactive agent enters (76->100%)
run src.eval.theta_sweep_readout --seeds 5         # THETA-SWEEP TOKENS ARE LOAD-BEARING (readout/LLM side): in a NOVEL per-episode layout a readout predicts the hazard AHEAD from real sweep tokens (90%) but collapses to chance when they are ablated or wrong-heading-shuffled; TrajectoryLLM(use_theta_sweep=True) feeds these (full frozen-LLM ablation: notebooks/m7_theta_sweep_llm_kaggle.py)
run src.eval.egocentric_anchors --seeds 5         # COEXISTING EGOCENTRIC ANCHORS (Nat Commun 2025): egocentric center + object + boundary frames coexist (decode all from the combined population), each specific to its own cells; adds EgocentricCenterCells
run src.eval.local_3d_order   --seeds 5            # LOCAL 3D ORDER not a global lattice (bat MEC): a local-order (blue-noise) field code has high local order but low global lattice -- separable from a true 3D lattice and from random
run src.eval.agent_grid_reanchor --seeds 5         # OBJECT REANCHORING IN THE CORE CORTEX: object-vector cells reanchor the grid phase from INSIDE _HexGridModules.forward(object_obs=) via a shared ego->allo transform; object cue rescues open-field drift where boundaries can't; shuffled-anchor control fails (load-bearing integration)
run src.eval.grid_3d          --seeds 5            # BIOLOGICALLY-GROUNDED 3D GRID CODE (bat MEC; Ginosar 2021): LocalOrder3DGrid wired into _HexGridModules(grid_3d=True) replaces the 1D-z stub -- fields in the bat regime (local order, NO global lattice) vs a cubic-lattice control; path-integrates + localizes in full 3D
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
echo "CAPSTONE: notebooks/m5_deadreckoning_llm_kaggle.py -- a frozen LLM reads the emergent HD+grid neural"
echo "  code to answer self-localization (WHERE + egocentric HOME vector); cortex-ON vs text-only-OFF, n=6."
echo; echo "DONE — see results/*.json and results/*.svg"
