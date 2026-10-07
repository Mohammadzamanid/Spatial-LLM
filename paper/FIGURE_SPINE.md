# Spatial-LLM flagship figure spine

This file fixes the main-paper narrative after completion of the two scientific submission gates.
Do not add new main-text experiments unless they replace a weaker panel.

## One-sentence paper claim

A structured cognitive-map interface gives a frozen language model causal spatial state that text alone lacks; the useful inductive biases are regime-dependent, replicate across LLM backbones, and transfer to real human trajectories.

## Figure 1 — System + causal language transfer

**Question:** Does the language model actually use the spatial cortex?

Panels:
- 1A: architecture schematic: self-motion → constrained grid/cognitive-map code → gated fusion → frozen LLM + LoRA.
- 1B: cortex ON vs OFF on the strongest single-item language readouts (torus and elapsed time).
- 1C: organ-specific double dissociation (WHERE: grid lesion; FACING/LANDMARK: corresponding organ lesion).

Main message: spatial answers are carried causally through the latent cortex, not prompt leakage.

## Figure 2 — What the grid code does and does not buy

**Question:** Is a grid code uniquely necessary?

Panels:
- 2A: Euclidean path integration — grid vs NoPE+sum certified tie.
- 2B: length extrapolation vs bounded place code.
- 2C: regime/phase diagram summarizing wins, ties, and losses.
- 2D: explicit negative results (sample-efficiency loss; noise tie).

Main message: the paper is not a universal grid-superiority claim; additive integration explains ordinary Euclidean PI, while bounded periodic population coding matters in specific regimes.

## Figure 3 — Periodicity and remapping as load-bearing mechanisms

**Question:** Where does the neural-style code become necessary/useful?

Panels:
- 3A: toroidal world — periodic code remains accurate across wraps while non-periodic additive baselines collapse.
- 3B: one-shot multimap memory — remapping prevents cross-context collisions.
- 3C: fixed-memory capacity / pattern-separation result.
- 3D: context-labelled trained-model control showing remapping advantage disappears when an external label substitutes for internal context.

Main message: identifiable representational mechanisms predict when the code helps.

## Figure 4 — Prospective theta-sweep + second-backbone replication

**Question:** Can the map provide prospective information, and is the effect backbone-specific?

Panels:
- 4A: theta-sweep mechanism schematic.
- 4B: Qwen2.5-1.5B ON / OFF / NO-SWEEP / wrong-heading across 8 seeds.
- 4C: SmolLM2-1.7B replication across 8 seeds.
- 4D: paired per-seed deltas for the preregistered ON−NO-SWEEP contrast.

Main message: the strongest causal prospective effect replicates across two open-weight LLM families.

## Figure 5 — External validation on GeoLife

**Question:** Does the fixed representation retain useful metric information on real trajectories from unseen people?

Panels:
- 5A: protocol schematic: GeoLife GPS → east/north steps → user-disjoint train/val/test → fixed grid population.
- 5B: 8-way bearing accuracy: GRID 76.6%, RAW-MLP 98.7%, OFF 16.2%.
- 5C: 6-bin distance accuracy: GRID 92.3%, RAW-MLP 98.8%, OFF 18.4%.
- 5D: held-out-user paired GRID−OFF effects with bootstrap CIs (bearing +59.3 pp; distance +76.5 pp; both p≈1×10^-5).

Main message: the representation-level result survives a real-world, user-disjoint external dataset without architecture changes.

## Supplementary / follow-up material

Keep these out of the main narrative unless needed to answer a reviewer:
- basal-ganglia action selection;
- 3D local-order/bat-grid extensions;
- boundary/object reanchoring details;
- content-binding what/when;
- semantic warp;
- replay;
- extra anatomical microcircuit modules;
- additional capacity/catastrophe diagnostics beyond the one panel needed for mechanism.

These remain valuable evidence and future-paper seeds, but putting them all in the main manuscript dilutes the causal cognitive-map story.

## Main-text stop rule

Every main-text result must serve at least one of:
1. causal transfer to language;
2. mechanistic boundary/necessity;
3. cross-backbone replication;
4. external validity.

Everything else belongs in Supplementary Information or a follow-up paper.
