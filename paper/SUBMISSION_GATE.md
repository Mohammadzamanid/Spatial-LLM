# Spatial-LLM submission gate — October 2026

This file is the publication stop rule. Do not expand the architecture until these gates are closed.

## Scientific claim

**Primary claim:** a structured cognitive-map channel can provide a frozen language model with causal spatial state that is absent from text, and different neural-style subcodes support dissociable spatial computations.

**Secondary claim:** the value of a grid-like population code is regime-dependent. It ties simpler additive integrators on ordinary Euclidean path integration, but becomes useful when bounded periodicity, remapping, fixed-memory pattern separation, or specific prospective computations are load-bearing.

**Not claimed:**
- that hexagonal grid geometry emerges from the constrained publication model;
- that grid codes universally outperform simpler sequence models;
- that the present controlled environments establish real-world embodied competence;
- that results on Qwen2.5-1.5B automatically generalize to other LLM families.

## Evidence already sufficient

- Multi-seed representation baselines and explicit null result: grid vs NoPE+sum is a certified tie on Euclidean path integration.
- Causal cortex-ON vs cortex-OFF language readouts.
- Torus readout: ON > OFF in every seed, paired p=0.033.
- Elapsed-time readout: ON > OFF in every seed, paired p=0.033.
- Allocentric/egocentric organ-specific lesion double dissociation.
- Theta-sweep language ablation: ON > no-sweep, shuffled, and OFF in every one of 8 seeds, p=0.0081.
- Raw JSON artifacts, figures, reproducibility script, and tests are committed.
- Negative and failed results are retained in the record.

## Remaining scientific blocker for a flagship submission

### Gate A — independent evaluation

Run the cognitive-map interface on **Microsoft GeoLife GPS Trajectories**, an external real-world trajectory dataset that matches the model's current self-motion/path-integration modality. The adapter is committed at `src/data/geolife_trajectory.py` and uses user-disjoint train/validation/test splits. Do not force-fit a multiview/video benchmark unless a visual front-end becomes part of the scientific question.

Locked protocol (pre-specified before the external run):
1. Dataset: official Microsoft GeoLife GPS Trajectories 1.3; non-overlapping windows of T={8,16,24}; maximum 4 windows per source trajectory/length.
2. Split: seeded-random **user-disjoint** 70/15/15 split (seed 20261007). No user appears in more than one split.
3. Scaling: compute median GPS step length from TRAIN users only and map it to 0.50 model units, matching the original synthetic speed regime. No validation/test scale tuning.
4. Tasks: primary = 8-way endpoint bearing; secondary = 6-bin endpoint distance with quantile cut-points estimated from TRAIN users only.
5. Conditions: fixed 6-module constrained GRID population; RAW exact additive endpoint displacement with a matched nonlinear readout; OFF train-majority predictor. RAW is a calibration baseline, not a straw-man target to beat.
6. Optimization: 8 readout seeds quantify training variance. **Held-out users, not seeds, are the inferential units**; primary significance is the paired GRID−OFF user-level sign-flip test plus a cluster/bootstrap CI over held-out users.
7. Success criterion for Gate A: GRID must exceed OFF on the primary bearing task with a user-level paired p<0.05 and a 95% user-bootstrap CI excluding zero. RAW performance determines how much information is lost relative to exact additive integration; GRID is not required to beat RAW.
8. Commit the exact script, result JSON, user-level effects, and aggregate statistics.

A visual benchmark such as MindCube or VSI-Bench is valuable related work, but using it directly would require adding a visual scene encoder and would test a different system.

### Gate B — backbone replication ✅ PASSED

Repeat the strongest causal language experiment on at least one second open-weight language model family or materially different scale.

Preferred experiment: **theta-sweep blocked-ahead** because it has the cleanest load-bearing token ablation.

Required conditions:
- same frozen cortex and dataset;
- same ON / OFF / no-sweep / wrong-heading controls;
- model-specific hyperparameters chosen without looking at test accuracy;
- at least 6 seeds, preferably 8;
- report all seeds and convergence diagnostics.

The claim needed is not identical absolute accuracy. The required replication is the **direction and causal specificity** of the spatial-channel effect.

**Completed on SmolLM2-1.7B-Instruct (n=8, forced-choice evaluation, 1600 steps):** ON 69.9% ±12.6 vs NO-SWEEP 49.8% ±0.3, Δ=+20.1 points, 7 wins / 0 losses / 1 tie, exact paired sign-flip p=0.0156. ON also exceeded text-only OFF by +19.9 points (p=0.0156) and wrong-heading sweep by +14.4 points (p=0.0156). This closes Gate B.

## Engineering / release gate

- [x] Publication branch created: `submission-ready-2026-10`
- [x] Hexagonal-emergence wording corrected in README and manuscript
- [x] Root MIT LICENSE added
- [x] Single-seed semantic-warp test made less brittle while preserving the mechanistic assertion
- [ ] CI green on Python 3.11 and 3.12
- [ ] Paper figure panels assembled
- [ ] References converted to a complete bibliography
- [ ] Release tag created after acceptance of the final artifact set
- [ ] Archive release on Zenodo / DOI
- [ ] Exact software citation updated to the release DOI

## Submission stop rule

A flagship submission is ready when Gate A is complete, CI is green, and the paper contains only claims directly supported by the committed artifacts. Gate B is complete.

Do **not** delay submission to add astrocytes, basal ganglia, additional plasticity mechanisms, or other modules unless a reviewer or a pre-registered hypothesis requires them. Those are follow-on papers.
