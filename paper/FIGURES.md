# Figure plan — main figures + Extended Data

A proposed, consistent figure set for the paper, mapping each committed `results/*.svg` to a numbered
figure with a one-line caption and the section it supports. The paper body currently cites Figures
1/5/6/7/8/9/10/11 (sparse and out of order — Fig 9 appears before Fig 6 in the text); this plan is the
target numbering to reconcile against (see "Reconciliation" at the end).

## Main figures (the narrative)

| # | source | caption (one line) | §  |
|---|---|---|---|
| **1** | `extrapolation.svg` | Length generalization: a scale-free grid cortex extrapolates distance/bearing to path lengths 3× the training range, vs place/GRU/oracle baselines. | §3 |
| **2** | `ablations.svg` + `seq_baselines.svg` | It is the inductive bias, not the architecture: range/scale/training-dist ablations, and the certified tie with a permutation-invariant NoPE+sum Transformer that shares the additive-integration bias. | §4 |
| **3** | `phase_diagram.svg` | When each inductive bias wins: regime × code matrix (win/tie/lose) — grid wins on cyclic worlds & one-shot capacity, ties where a plain integrator suffices, loses only at very low data. | §5 |
| **4** | `code_necessity.svg` + `frontier_probes.svg` | Where the population code is *necessary*: fixed-capacity associative memory + remapping (the sharp win); sample-efficiency/noise are honest non-wins. | §5 |
| **5** | `torus.svg` | Non-Euclidean necessity (CPU): on a torus the periodic code computes ∫v mod 2π at the oracle floor where additive integrators collapse to chance — a world a language prior cannot fake. | §5 |
| **6** | `emergence_gridcells.svg` | Emergent grid cells: periodic multi-field rate maps emerge in a cortex pretrained only on non-periodic place-cell prediction (measured, not designed). | §(emergence) |
| **7** | `successor.svg` | The predictive map (Successor Representation): plans detours around barriers where a Euclidean metric stalls (100% vs 62%, p=0.0086); fields track geodesic not Euclidean distance; grid-like SR eigenvectors; TD-learned. | §7 |
| **8** | `time_cells.svg` | Emergent time cells + scalar (Weber) timing: a generic recurrent substrate trained only to report elapsed time grows time cells (denser-early) whose fields widen with latency, yielding Weber-scalar timing — vs an untrained control. | §7 |
| **9** | `torus_llm.svg` + `elapsed_time_llm.svg` | Both codes read from language by a frozen LLM: torus-cell (space) and elapsed-time (time), each cortex-ON ≫ text-only-OFF, significant at n=6 (paired p=0.033); the queried quantity never in the prompt. | §8 |
| **10** | `structural_transfer.svg` | Abstract relational inference (TEM): a space-trained, frozen metric supports transitive inference + schema transfer on a non-spatial concept axis; falsified by shuffling the metric (p=0.009). | §6 |

(`extrapolation_llm.svg` — the grid-vs-place language extrapolation, n=3 — is a panel of Fig 9 or moves to
Extended Data; it is the one inconclusive-at-n=3 result and is *not* load-bearing for the framing.)

## Extended Data (supporting, referenced but not main)

| ED | source | what it shows |
|---|---|---|
| ED1 | `architecture.svg` | System schematic: cortex → spatial/temporal tokens → gated fusion → frozen LLM. |
| ED2 | `significance.svg` | Paired-significance table: every headline claim with sign-flip permutation p + bootstrap CI + Cohen's d. |
| ED3 | `controls.svg` | Mechanism-vs-parameters control (saturation is the axis, not param count). |
| ED4 | `multimap_task.svg` | Remapping boundary: it does NOT help a trained model that already has an external context label. |
| ED5 | `emergence_gridcells_hex.svg` + `emergence_gridcells_hexvel.svg` | Twisted-torus prediction tested: the square sheet does not flip to hexagonal firing (a clean falsification) though the twist improves the code. |
| ED6 | `planning.svg`, `goal_navigation.svg`, `continual.svg`, `embodiment.svg`, `boundary_anchoring.svg`, `pillars_hebbian.svg` | Integrative-substrate pillars: one code serves planning, dopamine value, one-shot/continual memory, vision-grounded path integration, boundary error-correction, Hebbian recall. |
| ED7 | `relational.svg` | Relational inference detail (symbolic-distance effect) backing Fig 10. |

## Reconciliation (in-text Figure numbers → this plan)

The body currently has these refs; update to the target scheme above when assembling:
- "Figure 1" (§3 extrapolation) → **Fig 1** ✓ (unchanged).
- "Figure 5" (§8 extrapolation_llm) → panel of **Fig 9** / ED.
- "Figure 6", "Figure 7" (torus CPU / torus_llm) → **Fig 5** (CPU torus) and **Fig 9** (torus_llm).
- "Figure 8" (structural_transfer) → **Fig 10**.
- "Figure 9" (phase_diagram) → **Fig 3**.
- "Figure 10" (successor) → **Fig 7**;  "Figure 11" (time_cells) → **Fig 8**.
- Add refs for **Fig 2** (mechanism), **Fig 4** (necessity), **Fig 6** (emergent grids), and the
  `elapsed_time_llm.svg` panel of **Fig 9** (currently uncited).

This renumbering is a single mechanical editing pass over the section bodies; deferred until the prose
pass so figure numbers and section text move together.
