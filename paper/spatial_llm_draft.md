# A grid-cell code gives language models length-generalizing spatial reasoning

**Working draft — Spatial-LLM.** Status markers: ✅ = result in hand, multi-seed; ⏳ = experiment
specified and running (Kaggle); ✎ = prose to finalize once ⏳ lands. Every number here is reproduced
by a script in `src/eval/` or a notebook in `notebooks/`; raw values in `results/*.json`.

---

## Abstract ✎

Large language models reason poorly about space, and what spatial competence they have does not
*extrapolate*: trained on short trajectories, they fail on longer ones. The mammalian brain solves
navigation with a specific representational scheme — velocity-driven **grid cells** that path-integrate
self-motion into a periodic, multi-scale metric, read out by **place cells**. We show that endowing a
frozen language model with this representation, learned self-supervised and with no coordinate labels,
transfers a spatial competence that **generalizes to trajectories several times longer than training**,
where the representations a conventional model would use *by default* — bounded place codes, standard
Transformers, length-normalized accumulators — collapse. With multi-seed, error-barred controls we trace
this to a specific **inductive bias** — additive, scale-free, order-invariant integration — that the
defaults lack and the grid code has *by construction* (its phase is a linear, un-normalized function of
integrated velocity, on a periodic multi-module lattice giving large metric range). We report honestly
that a sequence model *hand-built* with that same bias (a no-positional, sum-pooling Transformer)
extrapolates too — so the contribution is not that grid cells are the best path-integrator, but that they
*identify and embody* the right bias while being a single self-supervised biological code that
simultaneously serves higher cognition. The same code,
with no retraining of its metric, additionally supports vector-based **planning**, dopamine-like
**value** learning and goal navigation, and **relational/transitive inference** — evidence that a single
brain-faithful representation is a general substrate for spatial and relational cognition in a language
model.

---

## 1. The gap ✎

Coordinate embeddings let a model memorize a map; they do not let it *compute* over space in a way that
survives a change of scale. The decisive test of whether a system learned the **operation** of path
integration (rather than calibrating to a training length) is length extrapolation: train on short
walks, test on longer ones. Conventional sequence models and population codes fail this test for
identifiable reasons; grid cells do not, which is presumably why evolution converged on them.

Our thesis is concrete and falsifiable: a self-supervised grid-cell code is the representation that
makes spatial reasoning extrapolate, and that advantage transfers into a language model.

## 2. The representation extrapolates — isolated, multi-seed, fair baselines ✅

We strip away the language model and test the representation directly (`src/eval/extrapolation.py`). An
agent random-walks in 2-D (faithful to the language task's data); displacement grows ~√T, so longer
paths reach larger displacements. A position readout is trained on mixed short lengths {6,8,10,12}
(scale-free, no `/T`) and tested out to 4× longer; the three trajectory-QA tasks are derived from the
single decoded displacement. Four representations get identical data, training, and readout capacity —
only the code differs — and the **place baseline is tiled exactly to the trained region** (cells where
the model has been; longer paths reach beyond), the honest extrapolation question. *(An over-sized place
grid that pre-tiles the test range hides the effect — a trap an early draft of our own script fell into,
and which we flag explicitly.)* Mean ± 95% CI, n = 8 seeds:

| distance exact-acc | T=8 | T=16 | **T=24 (3×)** | T=48 (4×) |
|---|---|---|---|---|
| **grid code (ours)** | **99% ±0** | **97% ±1** | **93% ±0** | **75% ±0** |
| place tiling (trained region) | 97% ±0 | 90% ±0 | 80% ±1 | 57% ±1 |
| learned GRU integrator | 97% ±2 | 93% ±4 | 84% ±5 | 56% ±6 |
| exact-integration oracle | 99% | 99% | 99% | 99% |

The grid code wins at every length on every task with **non-overlapping CIs vs the place code**; the
bounded place population *cliffs* once paths leave its trained box (position error 0.047→0.594→1.787)
while the grid code degrades *gracefully* (0.017→0.174→1.041). The oracle is flat, so the gap is the
**code**, not the task. We do not overclaim: the grid code itself falls to 75% at 4× — its range is
large but finite. (Figure 1: `results/extrapolation.svg`.)

## 3. Why it extrapolates — the mechanism dissected ✅

Single-variable ablations, each multi-seed (`src/eval/ablations.py`, `seq_baselines.py`; n = 5):

1. **Metric range comes from modular coding.** A single periodic module aliases almost immediately (22%
   at 3×); adding modules at geometric scales lifts 4× accuracy monotonically 14%→82% (1→8 modules)
   (Fiete; Stensola 2012). This is why the grid code spans a large range with a fixed cell budget where
   a place tiling cannot.
2. **Scale-invariance is necessary.** A scale-free cumulative-sum readout is flat at 99% across length;
   dividing the same sum by path length T (the `/T` length-normalization) discards magnitude and
   collapses to 2% at 4×. The grid code is scale-free by construction (phase = gain·∫v).
3. **The advantage is in the code, not the training mix.** The grid code extrapolates even trained on a
   *single* length (69–77% at 4×); mixed-length training adds little — a sharp contrast with a learned
   accumulator, which *requires* mixed lengths to generalize.
4. **What sequence models reveal: it is the inductive bias, not the architecture.** Fed the same moves
   and budget (`seq_baselines.py`), the *default* Transformer (learned absolute positions, mean-pool)
   collapses (95%→16% by 3×, 2% at 4×); sinusoidal positions only partly help (38% at 3×); but a
   **NoPE + sum-pool** Transformer — permutation-invariant (correct for a commutative path sum) and
   additive — extrapolates *as well as the grid code at 3× and better at 4×* (88% vs 75%). A GRU is
   mediocre and seed-unreliable (82% ±8). **We report this against-us result prominently:** length
   extrapolation is not impossible for sequence models, it requires the *additive, scale-free,
   order-invariant integration bias* — which the conventional **defaults lack** and the grid code has by
   construction. (And a non-periodic additive integrator can exceed the finite-range grid code at extreme
   range; the grid code is not the best pure path-integrator — see §5 for what it uniquely provides.)

(Figures: `results/ablations.svg`, `results/seq_baselines.svg`.)

## 4. The advantage transfers to language ⏳

A LoRA-adapted Qwen2.5-1.5B answers navigation questions *through the frozen cortex* (the prompt holds
only the question; the moves reach the model only through the spatial code). Single-seed, the
grid-cell cortex beats the place/default cortex on every task and stays flat to 3× training length,
while cortex-OFF sits at chance (`src/training/train_trajectory.py`):

| task (cortex ON, T=8/16/24) | grid-cell cortex | place/default |
|---|---|---|
| return | 100 / 100 / 100 | 96 / 89 / 86 |
| bearing | 85 / 83 / 80 | 71 / 78 / 73 |
| distance (exact) | 95 / 88 / 85 | 62 / 46 / 40 |

⏳ **In progress:** the multi-seed version of this table (grid vs place × {distance,bearing} × seeds,
with 95% CI and the OFF control) is running on Kaggle T4
(`notebooks/m2_extrapolation_multiseed_kaggle.py`) and is the language-level counterpart to Figure 1.
This is the result that elevates §4 from illustrative to publication-grade.

## 5. The same code is a general cognitive substrate ✅

With its metric fixed, the grid code supports — multi-seed, mean ± 95% CI (`src/eval/stats.py`):

- **Planning** (Tolman novel-shortcut from the map): direction error **0.34° ± 0.04**, 100% navigable.
- **Value / goal navigation** (dopamine-like TD on the map): **95% ± 5** vs a random walker 29% ± 3.
- **Relational / transitive inference** (TEM-style, trained only on adjacent pairs): **84% ± 1** on
  never-seen non-adjacent pairs; a clean symbolic-distance effect (corr 0.96 ± 0.01).
- **One-shot / continual** (CLS): one-shot Hebbian recall **94% ± 2** vs a gradient baseline that
  catastrophically forgets (28% ± 5).

These are not the central claim, but they are strong evidence that the representation is *general* — one
brain-faithful code, many cognitive functions — which is the broader significance for AI.

## 6. Related work ✎

Grid cells and path integration (Hafting 2005; Burak & Fiete 2009); grid codes emerging in trained
path-integrators (Banino 2018; Cueva & Wei 2018); modular/periodic coding for range (Fiete; Stensola
2012); the Tolman-Eichenbaum Machine and grid codes in concept space (Whittington 2020; Constantinescu
2016); Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995); length generalization in
sequence models (a known-hard problem motivating positional-encoding research). Our contribution is to
(i) show the representational advantage *causally* with fair, multi-seed, error-barred controls, and
(ii) transfer it into a frozen LLM as length-generalizing spatial reasoning.

## 7. Limitations (honest) ✎

- The representation-level task is 2-D and uses an unbiased random walk; magnitude grows only ~√T, so
  "3× longer path" is ~1.7× larger displacement — the regime is modest and we say so.
- The grid code's range is finite (75% at 4×); we claim graceful, large-range extrapolation, not
  unbounded.
- The language results are at 1.5B parameters, LoRA, single-T4 scale; the multi-seed/​baseline version
  (§4 ⏳) is required before strong claims.
- Embodiment uses a simplified panoramic landmark world and a learned MLP front-end, not pixels.
- Boundary anchoring, replay, and remapping pillars are demonstrations, not the central claim.

## 8. Methods (brief) ✎

Velocity-driven hexagonal grid modules (`_HexGridModules`): fixed velocity gains at geometric scale
ratios integrate self-motion into a phase wrapped on a hexagonal torus; a learned linear readout maps
the grid-cell population to downstream features (entorhinal→hippocampal flow). Self-supervised
pre-training predicts bounded place-cell activity; no coordinate labels. The LLM path uses gated
cross-attention from the cortex into a frozen Qwen2.5-1.5B + LoRA. Full configs in `results/*.json`.

---

### Path to submission
- ⏳ §4 multi-seed LLM table + language-level figure (Kaggle, running).
- ✅ §2 Figure 1 (representation extrapolation), §3 ablations + fair sequence baselines, §5 stats — all
  multi-seed, committed.
- ✎ tighten abstract/intro/related-work; assemble figures; write Methods/Extended Data.
- Honest framing locked in §3.4 / abstract: the result is *which inductive bias matters* + the grid code
  as a unified substrate, not "grid cells beat everything."
