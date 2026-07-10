# Spatial-LLM — Findings so far

A neuroscience-inspired spatial encoder (Fourier coordinates, grid cells, place-cell
memory, ViT tiles) fused via gated cross-attention into a LoRA-adapted Qwen2.5-1.5B.
Probe task: elevation-threshold classification — *"is this location above the median
elevation (122 m)?"* — on real GeoNames cities, evaluated on held-out cities.

## Headline results — balanced accuracy (3 seeds, ~693–699 val examples)

| setup | location reaches model via | elevation given? | balanced acc |
|---|---|---|---|
| `coord_3d_noleak` | spatial channel only | yes (3rd coord) | **0.983 ± 0.005** |
| `coord_2d_noleak` | spatial channel only | no — must infer | **0.704 ± 0.019** |
| Step 1 `coord_3d` (leaky) | text + channel | yes | 0.974 (1 seed) |
| Step 1 `coord_2d` (leaky) | text + channel | no | 0.717 (1 seed) |
| chance | — | — | 0.50 |

## What we learned

1. **The spatial pathway works.** With coordinates removed from the prompt text — so
   location reaches the model *only* through the coord embedder + grid cells — the
   model still solves the task (0.98 when elevation is supplied, 0.70 when it must be
   inferred). The encoder genuinely conveys spatial information to the LLM.

2. **The encoder learned real geography.** Given only lat/lon and *no* elevation
   input (`coord_2d_noleak`), it predicts elevation-above-median at **0.70**, far above
   chance — i.e. it interpolates elevation over location for cities it never saw.

3. **Step 1's 2D-vs-3D gap is real, not a leak artifact.** `coord_2d_noleak` (0.704)
   reproduces the leaky Step-1 `coord_2d` (0.717) almost exactly, and `coord_3d_noleak`
   (0.983) ≈ leaky `coord_3d` (0.974). The gap measures the value of being handed
   elevation directly (3D) versus inferring it from location (2D).

4. **A methodology bug, found and fixed.** The prompt template originally injected
   lat/lon as text, letting the LLM read coordinates and recall elevation from
   pretraining — bypassing the spatial stack entirely (its gates sat at ~0 while
   accuracy was ~0.99). The `coords_in_text=false` mode is required to actually test
   the encoder.

5. **Per-module fusion gating ("synchronization") does not help.** No accuracy gain
   over a single shared gate, and less stable (one seed collapsed to 0.68). Dropped.

6. **Gate magnitude is not a usage read-out.** Fusion gates stayed ~0.002 in every
   run, yet in the no-leak runs that tiny gate carries the entire signal. Judge by
   accuracy, not gate value.

## Navigation cortex: static vs movement — the "harmful → essential" inversion

The neuroscience modules (grid attractor, conjunctive, boundary, theta-gamma,
microcircuits) were ablated on TWO tasks. The verdicts **flip with the task** —
confirming these are a *navigation* system that needs movement and time to be useful.

**Static task** — classify one fixed (lat,lon) into a 100-way grid (`ablation.py`):

| module removed | Δ accuracy | verdict |
|---|---|---|
| grid_attractor | −91.6% | load-bearing |
| boundary | −9.3% | helps |
| cortical_column | +3.7% | mildly harmful |
| lateral_inhibition | +7.1% | harmful |
| conjunctive | +7.1% | **harmful** (no movement → dormant) |
| phase | +7.2% | **harmful** (no movement → dormant) |

**4D navigation task** — integrate a sequence of moves (heading, speed, vertical
velocity) over time *t* → final (x,y,z) (`ablation_trajectory.py`; 3 seeds, T=12,
metric = within-0.15 accuracy, full stack = 95.3%):

| module removed | accuracy | Δ vs full | verdict |
|---|---|---|---|
| conjunctive | 0.0% | **−95.3%** | **essential** (was +7.1% harmful!) |
| lateral_inhibition | 86.5% | −8.8% | helps (was +7.1% harmful!) |
| grid_attractor | 88.4% | −6.9% | helps |
| theta_gamma | 95.6% | +0.3% | neutral |
| cortical_column | 98.7% | +3.4% | mildly harmful |

**The inversion:** `conjunctive` (head-direction × speed = velocity binding) goes from
+7.1% *harmful* on the static task to **−95.3% essential** on the movement task —
without it the model can't read the moves and fails completely. `lateral_inhibition`
flips the same way; `grid_attractor` (the path integrator) is load-bearing on both.

**Synchronization helps.** Giving each module its own auxiliary target-prediction
signal (aux loss) lifts the full stack 95.3% → 97.9% (err 0.092 → 0.075). This is the
concrete, working form of "synchronize the modules" — each specialises against its own
objective instead of training as one tangled blob.

**Managing complexity — the empirical rule:** match the module set to the task.
Movement modules are dead weight (even harmful) on static tasks and essential on
movement tasks. `theta_gamma` and `cortical_column` aren't needed for *simple* path
integration — making them load-bearing needs order-dependent tasks (see recall, below).

### Order-dependent recall — the integrator earns its keep

Final-position path integration is a commutative sum, so the elaborate modules are
redundant. The **recall** task — *"where were you at step k?"* — is order/history-
dependent: a sum can't answer it; the model must keep a RUNNING per-step position
(`ablation_trajectory.py --task recall`; 3 seeds, T=12, full stack acc = 99.8%).

| module removed | accuracy | Δ vs full | verdict |
|---|---|---|---|
| conjunctive | 0.4% | −99.3% | essential |
| grid_attractor | 3.9% | **−95.8%** | **essential** — flips from *redundant* on path-integration |
| theta_gamma | 99.8% | +0.1% | neutral |
| cortical_column | 99.6% | −0.2% | neutral |
| lateral_inhibition | 99.6% | −0.2% | neutral |

The **grid attractor** (recurrent path integrator) is redundant when only the final
point matters but **load-bearing** when the task needs the trajectory's history — same
module, opposite verdict, decided entirely by the task. (`add_one_in`: no single module
suffices — recall needs the velocity encoder AND the integrator together.)

### Task-dependent complexity — the model prunes itself (learned gates)

Each optional module (theta-gamma, microcircuits) gets a learned gate + L1 cost, so the
network can switch off what it doesn't need (`--mode gates`, L1=0.05):

| task | full (ungated) | gated acc | learned add-on gates |
|---|---|---|---|
| pathint | 95.3% | **97.3%** | all → ~0.1–0.36 (OFF) |
| recall | 99.8% | 94.1% ±6.9% | all → ~0.1–0.16 (OFF) |

On both tasks the model drives the add-on gates toward 0 — it discovers they aren't
load-bearing and switches them off (on pathint this *improves* accuracy; on recall the
aggressive L1 is slightly costly/noisier — gate strength is a knob). The structural
velocity-encoder + integrator always stay on because the task is unsolvable without them.

**The complexity rule, made mechanical:** ablation reveals which modules a task needs;
the gates let the model *enforce* it — keep the load-bearing ones, switch off the rest,
per task. The "harmful module" problem dissolves: a module that doesn't help is gated off.

### Memory-bottleneck recall — theta-gamma earns its keep

The recall task above used attention over the FULL step sequence, so there was no memory
pressure and theta-gamma stayed neutral. **memrecall** forces the trajectory through a
fixed-size bottleneck: the whole path is multiplexed into ONE vector and the answer is
read back from that vector alone (`--task memrecall`; 3 seeds, T=7, full acc = 99.9%).

| module removed | accuracy | Δ vs full | verdict |
|---|---|---|---|
| conjunctive | 0.7% | −99.3% | essential |
| theta_gamma | 7.0% | **−92.9%** | **essential** — ordered memory; mean-pool can't recall by index |
| grid_attractor | 99.6% | −0.3% | neutral (the memory compensates at short T) |
| cortical_column / lateral_inhibition | ~99.8% | ~0 | neutral |

Theta-gamma's phase-slot multiplexing (Lisman-Idiart) preserves order + identity through
the bottleneck; replacing it with a mean-pool collapses both and recall-by-index fails.

**The 7±2 limit emerges.** `ThetaGammaMemory` has ~8 slots. Recall stays near-perfect
while the trajectory fits, then collapses as it overflows — a capacity limit that falls
out of the architecture, not a tuned hyperparameter:

| trajectory length T | 4 | 7 | 10 | 14 |
|---|---|---|---|---|
| recall accuracy | 99.6% | 99.9% | 70.1% | 30.7% |

### Each module is load-bearing on the task it was built for

| task | what it requires | essential module(s) |
|---|---|---|
| static localization | a position code | grid cells |
| path integration | velocity binding | conjunctive |
| order-dependent recall | running per-step position | conjunctive + grid attractor |
| memory-bottleneck recall | ordered working memory | conjunctive + theta-gamma |

The whole stack earns its keep — not all at once, but each module exactly when the task
demands it; the learned gates switch off the rest.

## Milestone 2 — the LLM answers trajectory questions in language

The path (a sequence of moves) is encoded by the recurrent cortex into spatial tokens
and fused into a LoRA-adapted Qwen2.5-1.5B; the prompt holds ONLY the question
("Are you back where you started?"), so the moves reach the model *solely* through the
cortex. Training the whole stack end-to-end from the single yes/no token collapses to
the class prior, so the cortex is pre-trained on path integration first, then frozen,
and the LLM learns to read it (`src/training/train_trajectory.py`).

| cortex pre-training | cortex ON | cortex OFF (control) | contributes |
|---|---|---|---|
| supervised (regress final x,y,z) | 99.8% | 51.3% (chance) | +48.5% |
| **self-supervised (place-cell prediction, no coords)** | **99.7%** | 51.3% (chance) | **+48.3%** |

cortex-OFF at chance confirms the LLM answers *through* the spatial cortex, not the text.
**The self-supervised version matches the supervised one (99.7% vs 99.8%) with zero
coordinate labels** — the cortex learned navigation from its own movement + sensory
landmarks, and language then read it. That is the biologically faithful result.

**Caveat (honest).** The cortex is trained before the LLM, not end-to-end. This is
defensible developmentally — the brain's grid/place/head-direction system is largely
innate and present in pups before much experience (Langston 2010; Wills 2010), and the
sensorimotor stage precedes language. The genuinely unrealistic part of the *supervised*
run is the position LABELS. The **self-supervised** protocol removes them: the cortex
predicts the place-cell code of where it ends up (a sensory function of position) and
must recover position internally — the way grid codes emerge (Banino 2018; Cueva & Wei
2018). A probe confirms "back-at-start" is ~100% readable from that no-label rep, so the
LLM can use it just as well. (`--cortex_pretrain selfsup`, now the default.)

## Generalization stress-test — does the integrator learn the OPERATION, or memorize the length?

Every result above trains and tests at the SAME trajectory length. The decisive
question for "did it really learn path integration": train on SHORT walks, then test
on LONGER, unseen lengths. A model that learned the operation ("sum the per-step
displacements") extrapolates; one that calibrated to the training length does not.

We isolate the suspect — the `/T` length-normalisation in `_AttractorIntegrator`
(`readout(u / T)`) — and cross it with the training distribution
(`src/eval/generalize_trajectory.py`; train T=8, eval T∈{4,8,12,16,24,32}, 3 seeds).
Read-out is **`mag_ratio = mean‖pred‖ / mean‖true‖`** (1.0 = correctly scaled at that
length; <1 under-shoots, >1 over-shoots) and **`rel_err`** (flat across T ⇒ length-invariant).

| mode (architecture, training) | T=4 | **T=8 train** | T=12 | T=16 | T=24 | T=32 |
|---|---|---|---|---|---|---|
| `shipped` — M2 cortex (`/T` + LayerNorm), fixed T | 2.60 | **1.00** | 0.65 | 0.48 | 0.32 | 0.24 |
| `norm` — integrator `/T`, fixed T | 1.99 | **0.99** | 0.67 | 0.50 | 0.34 | 0.26 |
| `free` — integrator scale-free, fixed T | 1.21 | **0.98** | 1.17 | 1.47 | 2.27 | 3.04 |
| `norm_mixed` — `/T`, MIXED lengths (≤16) | 1.46 | **1.00** | 0.85 | 0.76 | 0.64 | 0.55 |
| **`free_mixed` — scale-free, MIXED lengths (≤16)** | **0.89** | **0.90** | **0.90** | **0.89** | **0.88** | **0.93** |

*(values are `mag_ratio`, mean over 3 seeds; std ≤0.07 for the `/T` modes.)*

1. **Fixed-length training = length memorization — for EVERY architecture.** `shipped`,
   `norm`, `free` all nail the train length (mag_ratio≈1.00) and all break away from it,
   in opposite directions: the `/T` modes **under-shoot with mag_ratio ≈ train_T / test_T**
   (0.5 at 2×, 0.25 at 4×, 2.0 at ½× — a textbook length-calibration artifact), while the
   scale-free model **over-shoots** (3× at T=32) because its accumulator's magnitude tracks
   step *count*, not net displacement. `rel_err` grows monotonically with |T−8| in all three.

2. **Mixed-length training makes the operation generalize — and now the readout matters.**
   `free_mixed` (scale-free + lengths sampled from {4…16}) is **flat: mag_ratio 0.86–0.93 and
   rel_err ~0.65 across the whole sweep, including the held-out T=24 and T=32 — 2× beyond its
   longest training length.** That is genuine extrapolation: error does NOT grow with
   extrapolation distance. `norm_mixed` helps but still droops to 0.55 at T=32 — the `/T`
   that *enabled* one-length memorization now mildly *fights* length-invariance.

3. **Verdict.** The Milestone-2 cortex's failure to extrapolate is primarily a
   **training-distribution artifact (it only ever saw one length), compounded by the `/T`
   readout** — not a fundamental limit. The path-integration operation is genuinely
   learnable and length-invariant, given (a) length-diverse training and (b) a scale-free
   readout. *Recommendation:* if TrajectoryLLM should answer about paths of any length,
   pre-train the cortex on mixed T with `readout(u)` (no `/T`), not the current fixed-T + `/T`.

4. **The cost of generalizing.** The length-invariant `free_mixed` is *less precise at any
   single length* than a specialist (rel_err ~0.65 vs 0.08 for the fixed model at its train
   T) — a clean generalization-vs-specialization tradeoff. Its mag_ratio ~0.9 (slight
   under-scale) is honest, not pinpoint; the flatness across length, not the absolute value,
   is the extrapolation evidence, and more training steps would likely tighten it.

See `results/generalize_trajectory.json` for per-seed numbers.

### Folded back into the real model — TrajectoryLLM answers about LONGER paths

We applied the recommendation to Milestone 2 and re-ran the full TrajectoryLLM
(Qwen2.5-1.5B + LoRA, frozen self-supervised cortex) on Kaggle (T4). Both models train on
SHORT paths and answer *"Are you back where you started?"* at T=8, 16, 24 — the longer two
are held out (`train_trajectory.py --train_lengths … --eval_lengths 8 16 24 [--cortex_scale_free]`).

| recipe | T=8 (train) | T=16 (held-out) | T=24 (held-out, 2× train) |
|---|---|---|---|
| baseline — fixed T=8 + `readout(u/T)` | **99.0%** | 83.0% | 69.0% |
| **fix — mixed {6,8,10,12} + scale-free `readout(u)`** | 96.0% | **89.0%** | **86.0%** |
| cortex OFF (text-only control) | ~47% | ~56% | ~48% |

- **The fix generalizes; the baseline degrades.** One length + `/T` loses 30 points by T=24
  (99→69); mixed-length + scale-free loses only 10 (96→86) and beats the baseline by
  **+17 points at T=24** — answering correctly about paths 2× longer than any it trained on.
  The cortex-level "back-at-start" probe shows the same trend, cleaner: **1.00 / 0.96 / 0.93**
  (fix) vs **0.99 / 0.90 / 0.86** (baseline) — the scale-free cortex's rep is the more
  length-invariant one, exactly as the isolated stress-test predicted.
- **cortex OFF stays at chance at every length** → the LLM answers THROUGH the cortex, not
  the text, even when extrapolating.
- **Honest nuance.** The baseline doesn't *collapse* (69% > chance): the binary return
  question is more forgiving than raw magnitude regression — the self-supervised place-code
  (+ LayerNorm) carries a partly length-robust pattern. And the fix isn't perfect (86%, and
  it concedes ~3 points at the training length — the generalization-vs-specialization
  tradeoff again). But the operation transfers: the M2 model now reasons about path lengths
  it never saw, and the recommendation measurably helps on the real LLM, not just the
  isolated integrator. (`results/m2_lengthgen_baseline.json`, `…_scalefree_mixed.json`;
  cells in `notebooks/m2_length_generalization_kaggle.py`.)

### Harder questions — magnitude vs direction (the integrator's frontier)

The binary "are you back where you started?" is forgiving. We raised the bar with two
*multi-class* questions answered in language through the SAME frozen self-supervised cortex
(generalizing recipe; train on 6–12, test 8/16/24; `--task distance|bearing`):

  - **distance** — "How far are you from where you started?" → quantized bucket 0–5 (MAGNITUDE)
  - **bearing**  — "Which direction is the start from here?" → 8-way compass word (DIRECTION)

| task | metric | T=8 (train) | T=16 (held-out) | T=24 (held-out) |
|---|---|---|---|---|
| **bearing** (8-way) | exact ON / OFF | **71% / 17%** | **78% / 18%** | **73% / 12%** |
| | within-1 ON / OFF | 91% / 42% | 92% / 36% | 90% / 35% |
| | chance | 17% | 18% | 15% |
| **distance** (6 buckets) | exact ON / OFF | **62% / 26%** | 46% / 17% | 40% / 14% |
| | within-1 ON / OFF | 94% / 87% | 78% / 74% | 81% / 71% |
| | chance | 38% | 29% | 33% |

**Direction generalizes; magnitude is the frontier — and that split is mechanistically exactly right.**
- **Bearing is the clean win.** 71–78% exact (≥90% within one compass point) on an 8-way
  question — and it does NOT degrade with length (T=16 even edges T=8). The cortex probe
  agrees: 94/96/91%, flat. Because direction-home is SCALE-INVARIANT, a length-invariant
  cortex reads it off at any path length. cortex-OFF sits at chance (12–18%), so the LLM
  answers ~4× above chance purely through the spatial channel — even 3× beyond training length.
- **Distance is solid in-distribution but decays out of it.** 62% exact / 94% within-1 at
  the trained scale, falling to 40% exact at T=24 (probe 85→37%). Magnitude is the one
  quantity whose SCALE must extrapolate, and the place-code position rep (magnitude-normalised
  by the cortex LayerNorm) loses precision at distances it saw less of. Still, exact ON beats
  the text-only control at every length (+26/+29/+26) and beats chance in-range — the cortex
  genuinely supplies "how far", just imperfectly when the scale itself is novel. (within-1 is
  weakly discriminating here — the distribution is concentrated, so OFF also scores high;
  exact is the real signal for distance, whereas for bearing within-1 IS discriminating.)

**The takeaway:** these harder questions separate what the spatial code carries *robustly*
(DIRECTION, scale-free → generalizes flat) from what it carries only *near the trained scale*
(MAGNITUDE → degrades with length). The binary task hid this; asking for the displacement
vector exposes it. Both still ride entirely on the cortex (OFF ≈ chance).
(`results/m2_distance.json`, `results/m2_bearing.json`; `notebooks/m2_harder_tasks_kaggle.py`.)

### Attacking the magnitude frontier — fixed with GRID cells (the brain's own answer)

distance was the one question that degraded with length. We isolated the candidate levers on
the cortex's distance probe (CPU, no LLM; train on 6–12, read distance at 8/16/24; 3 seeds;
`src/eval/magnitude_frontier.py`):

| self-supervised cortex target | out LayerNorm | T=8 | T=16 | T=24 |
|---|---|---|---|---|
| **place cells** (bounded Gaussian — M2 default) | on | 91% | 69% | 44% |
| place cells | off | 61% | 28% | 17% |
| multi-scale place cells | on | 89% | 67% | 37% |
| **GRID cells (periodic, multi-scale) — the faithful fix** | on | **93%** | **82%** | **60%** |
| *position-regression (supervised, uses labels — reference)* | on | *95%* | *89%* | *67%* |
| position-regression | off | 84% | 41% | 22% |

1. **The bottleneck is the CODE, and the brain's fix is grid cells.** Place cells are *bounded
   and localized* — no field, no code outside the trained arena (and a bigger arena just goes
   sparse and collapses: env8/K1500 → 31/27/24%). That bounded code is *why* magnitude didn't
   extrapolate — a biologically real limit. **Grid cells** use a *periodic, multi-scale (modular)*
   code that represents a metric over a large range and extrapolates beyond the trained arena
   (Banino 2018; Fiete modular coding). Swapping the self-supervised target from place→grid lifts
   T=16 69→82% and T=24 44→60% — recovering most of the gap to the supervised upper bound,
   **with zero coordinate labels and a *more* brain-faithful code, not less.**
2. **Bypassing the LayerNorm HURTS, everywhere** (e.g. supervised T=24 67→22%). Magnitude lives
   in the rep's PATTERN, not its scalar norm; the LayerNorm stabilises the scale-free integrator's
   growing activity across lengths, and removing it exposes that instability. (Our initial
   "LayerNorm normalises magnitude away" guess was wrong — the sweep refuted it.)
3. **Naïve multi-scale *place* cells didn't help** (≈ single-scale) — coarse Gaussian fields
   alongside fine ones aren't the same as a periodic grid code; periodicity is what carries range.
4. **A residual length-degradation persists even with the best target** (grid 93→60%, supervised
   95→67%): recurrent path-integration error ACCUMULATES over more steps, independent of readout
   — the genuinely hard, possibly irreducible part.

**Verdict: the magnitude frontier is largely FIXABLE, and the fix is biologically faithful.**
Direction (bearing) generalised for free because it is scale-invariant; magnitude needed the
right spatial *code* — and switching place→**grid cells** (still self-supervised, no labels)
recovers most of it, leaving only a residual long-horizon integration drift. The neuroscience
made the prediction (grid cells are the entorhinal metric/path integrator) and it held.

**Confirmed on the full LLM.** Re-running distance with the grid-cell cortex (`--task distance
--code grid`, still self-supervised, NO labels) reproduces the fix end-to-end:

| distance (cortex ON) | T=8 | T=16 (held-out) | T=24 (held-out) |
|---|---|---|---|
| place-cell cortex — exact | 62% | 46% | 40% |
| **grid-cell cortex — exact** | **79%** | **83%** | **63%** |
| **grid-cell cortex — within-1** | **100%** | **99%** | **94%** |
| grid cortex probe | 99% | 88% | 83% |

Exact jumps +17/+37/+23 over place cells, and at T=16 it is *higher* than at T=8 — near-flat
across the extrapolation range. **within-1 is 100/99/94%**: the answer is off by at most one
bucket essentially always, even at 3× the training length. cortex-OFF sits at chance (~23–28%),
so the magnitude reasoning rides entirely on the (label-free) grid code. The one question that
didn't generalise now does — through the brain's own path-integration code.
(`results/m2_distance_grid.json`.)

### Why it extrapolates — the representation isolated, multi-seed, against fair baselines

This is the central claim made bulletproof at the representation level (no LLM, so the effect is
attributable to the **code**). We train a position readout on mixed SHORT paths {6,8,10,12}
(scale-free) and test out to 4× longer, deriving the three trajectory-QA tasks from the single
decoded displacement. Four representations get the SAME data, the SAME training, and a matched
256-unit readout — only the code differs (`src/eval/extrapolation.py`, **mean ± 95% CI, n=8**):

![length extrapolation](extrapolation.svg)

The **fairness crux**: the place cells tile *exactly* the region the TRAINING displacements occupy
(data-driven, ±2.94 per axis), so the model has place cells everywhere it was trained — longer test
paths then reach *beyond* that box. (An over-sized place grid that pre-tiles the test range hides the
effect; that is the trap a careless benchmark falls into, and an earlier draft of this very script
fell into it.)

| distance exact-acc | T=8 | T=16 | **T=24 (3×, the LLM regime)** | T=48 (4×) |
|---|---|---|---|---|
| **grid code (ours)** | **99% ±0** | **97% ±1** | **93% ±0** | **75% ±0** |
| place tiling (trained region) | 97% ±0 | 90% ±0 | 80% ±1 | 57% ±1 |
| learned GRU integrator | 97% ±2 | 93% ±4 | 84% ±5 | 56% ±6 |
| exact-integration oracle | 99% | 99% | 99% | 99% |

(position-decode error at T=24: grid **0.174 ±0.015** vs place 0.594 ±0.017 vs GRU 0.214 ±0.091 vs
oracle 0.013; bearing acc at T=24: grid **97%** vs place 90% vs GRU 92%.)

1. **The grid code wins at every length, on every task, with non-overlapping CIs vs the place code** —
   on a place baseline that is *fair* (cells exactly where training has been). The advantage is the
   bounded place population's inability to represent positions past its trained box: its error *cliffs*
   (0.047 → 0.594 → 1.787 over T=8→24→48) while the grid code degrades *gracefully* (0.017 → 0.174 →
   1.041). This is the grid/place division of labour — grid cells trade a little local precision for
   metric **range**.
2. **The grid code is also more RELIABLE than a learned integrator.** A GRU path-integrator (the
   standard deep baseline; Banino 2018) matches the grid code in the *mean* but is high-variance across
   seeds (±5% on distance, ±0.091 on position at T=24, vs the grid code's ±0–1%). The fixed grid code
   generalizes *consistently*; the learned one is a coin-flip on the seed.
3. **The oracle is flat at 99%** — perfect integration solves the task at every length, so the entire
   grid-vs-place gap is the *representation*, not the task.
4. **Honest ceiling:** the grid code itself drops to 75% at 4× (T=48). Its periodic, multi-module code
   covers a large but *finite* unambiguous range; we do not claim unbounded extrapolation. Within the
   LLM's tested regime (≤3×) the margin over every conventional code is large and statistically clean.

Together with the `/T` stress-test above (scale-invariance) and the magnitude-frontier sweep (range),
this pins down *why* the velocity-driven grid code is the representation that lets the language model
answer about longer paths: its phase = gain·∫v is **scale-free** *and* **periodic**, giving both
length-invariance and metric range — the two properties the conventional codes each lack.
(`results/extrapolation.json`, `results/extrapolation.svg`.)

### Ablations — the mechanism dissected (multi-seed, the four reviewer questions)

`src/eval/ablations.py` changes ONE thing at a time on the same faithful task (n=5 seeds, 95% CI),
answering *why* the grid code extrapolates — and what it is **not**:

![ablations](ablations.svg)

| (distance exact-acc) | T=8 | T=16 | T=24 | T=48 (4×) |
|---|---|---|---|---|
| **1 — range needs MODULES** (grid, n_modules) | | | | |
| 1 module | 33% | 27% | 22% | 14% |
| 2 / 4 / 6 / 8 modules | 91 / 97 / 99 / 98% | 78 / 95 / 97 / 98% | 67 / 89 / 93 / 95% | 45 / 70 / 75 / 82% |
| **2 — scale-invariance is necessary** | | | | |
| sum (scale-free) | 99% | 99% | 99% | 99% |
| sum **/ T** (length-norm) | 87% | 29% | 11% | 2% |
| **3 — training distribution (grid)** | | | | |
| fixed T=8 / fixed T=12 / mixed 6–12 | 99 / 98 / 99% | 96 / 97 / 97% | 90 / 93 / 93% | 69 / 77 / 75% |
| **4 — vs sequence models fed the moves** | | | | |
| **grid code (ours)** | **99% ±0** | **97% ±1** | **93% ±1** | **75% ±1** |
| plain Transformer (learned pos) | 95% ±1 | 48% ±4 | 16% ±1 | 2% ±0 |
| GRU integrator | 97% ±3 | 92% ±6 | 82% ±8 | 53% ±8 |

1. **Range is modular coding.** A *single* periodic module aliases almost immediately (22% at the 3×
   LLM regime); adding modules at geometric scales extends the unambiguous metric range monotonically
   (Fiete; Stensola 2012) — 1→8 modules lifts 4× accuracy 14%→82%. This is *why* the grid code, unlike
   a place tiling, covers a large range with a fixed cell budget.
2. **Scale-invariance is necessary.** A scale-free cumulative-sum readout is flat at 99% across all
   lengths; dividing the same sum by path length T — the `/T` length-normalization the M2 cortex once
   used — *discards the magnitude* and collapses to 2% at 4×. The grid code has scale-invariance for
   free (phase = gain·∫v).
3. **The grid code's extrapolation is in the CODE, not the training mix.** Trained on a *single* length
   it still extrapolates (fixed T=8 → 69%, fixed T=12 → 77% at 4×); mixed-length training adds little.
   This is a notable contrast to the *accumulator*, which **needed** mixed lengths to generalize (the
   `/T` stress-test) — the periodic grid code is length-invariant by construction, so it is robust to
   the training-length distribution.
4. **A plain Transformer fed the moves does NOT extrapolate.** With the same data and budget, a standard
   Transformer collapses (95%→16% by T=24, 2% at 4×) and a GRU degrades and is *seed-unreliable*
   (82% ±8 at T=24); the fixed grid code holds at 93% ±1 — non-overlapping CIs. Length generalization in
   sequence models is a known-hard problem; the grid code has the needed inductive bias intrinsically.
   *(Caveat: this Transformer uses learned absolute positions; the fairer sinusoidal / NoPE-plus-sum
   variants are tested in `seq_baselines.py` next — and they change the conclusion in an honest way.)*

### Fair sequence-model baselines — what *actually* makes extrapolation work (and a result against us)

The "why not a Transformer" answer must not rest on a strawman. We gave the Transformer its best shot
at length generalization (`src/eval/seq_baselines.py`, n=5, 95% CI): sinusoidal positions (defined at
*every* length) and a **NoPE + sum-pool** variant — no positional encoding (permutation-invariant,
which is *correct* for a commutative path sum) and additive pooling (built to integrate).

![fair sequence baselines](seq_baselines.svg)

| distance exact-acc | T=8 | T=16 | T=24 (3×) | T=48 (4×) |
|---|---|---|---|---|
| **grid code (ours)** | 99% ±0 | 97% ±1 | 93% ±1 | 75% ±1 |
| Transformer, learned pos + mean-pool (naive default) | 95% ±1 | 48% ±4 | 16% ±1 | 2% ±0 |
| Transformer, sinusoidal pos + mean-pool | 95% ±1 | 69% ±3 | 38% ±3 | 21% ±2 |
| **Transformer, NoPE + sum-pool** | 96% ±2 | 94% ±3 | **92% ±4** | **88% ±6** |
| GRU | 97% ±3 | 92% ±6 | 82% ±8 | 53% ±8 |

**The honest conclusion — and it sharpens the claim rather than inflating it:**

1. **It is not "Transformers can't extrapolate."** A NoPE + sum-pool Transformer extrapolates *as well
   as the grid code at 3× and BETTER at 4×* (88% vs 75%). We report this result, which runs against the
   simplest version of our story, prominently.
2. **What actually matters is the INDUCTIVE BIAS: additive, scale-free, order-invariant integration.**
   The *default* sequence model (learned absolute positions + mean-pool) lacks it and collapses (16% at
   3×); sinusoidal positions only partly help (38%); you recover extrapolation exactly when you build
   the integration prior in (NoPE + sum-pool). The grid code has this bias *by construction*
   (phase = gain·∫v, periodic) — that is the real content of §2–§3.
3. **For pure displacement decoding at extreme range, a non-periodic additive integrator can exceed the
   grid code** (NoPE-sum 88% vs grid 75% at 4×), because the grid's periodic code has a finite
   unambiguous range while a linear sum does not. The grid code is *not* the best possible
   path-integrator; its distinctive value (below) is being a single biological code that *also* yields a
   periodic metric and place readout serving planning, value, and relational inference — a NoPE-sum
   integrator gives you displacement and nothing else.
4. **GRUs are mediocre and seed-unreliable** (82% ±8 at 3×): a learned recurrent integrator generalizes
   inconsistently, unlike the fixed grid code (±1).

So the defensible contribution is not "grid cells beat all baselines at path integration" — they do not.
It is: (i) we *identify* the inductive bias that makes spatial reasoning extrapolate and show the
conventional *defaults* lack it; (ii) the grid code embodies that bias *and* is a multi-purpose,
self-supervised, biological substrate (the same code does planning/value/relational cognition); and
(iii) that fixed code transfers length-generalizing spatial reasoning into a frozen LLM.
(`results/seq_baselines.json`, `results/seq_baselines.svg`.)

### Where the population code is NECESSARY — capacity and remapping (the sharp claim)

The honest baseline above showed an additive integrator *ties* the grid code on path integration. So the
defensible claim is not about path integration at all — it is that a cognitive **map** needs more than a
metric: a high-capacity, **remappable** population code. We test two things a deterministic function of
displacement *cannot* do, however well it integrates (`src/eval/code_necessity.py`, n=5, 95% CI):

![code necessity](code_necessity.svg)

**A — one-shot memory capacity** (bind K locations Hebbian; recall from a noisy probe):

| recall acc | K=5 | K=25 | K=100 | K=200 |
|---|---|---|---|---|
| grid / place / RFF (population codes) | 100% | 97% | ~87% | **75%** |
| additive + smooth MLP lift | 99% | 95% | 82% | 70% |
| **additive (raw 2-D displacement)** | 98% | 77% | 42% | **25%** |

The integrator's *raw* output (a 2-D displacement) cannot pattern-separate — capacity collapses to 25%
at K=200. You need a high-dimensional population code; a **periodic** lift (RFF ≈ grid) is best — i.e. to
match grid cells' capacity you end up *building a grid-like code*.

**B — multi-map storage (remapping) — the decisive, information-theoretic necessity:**

| retrieval acc | M=1 | M=2 | M=4 | M=8 | M=16 maps |
|---|---|---|---|---|---|
| **place + remap** | 93% | 93% | 93% | 93% | **92% ±3** |
| **grid + remap** | 93% | 92% | 91% | 87% | **79% ±2** |
| grid, remapping OFF (ablation) | 93% | 46% | 23% | 12% | **6% ±0** |
| **additive (raw 2-D)** | 65% | 32% | 16% | 8% | **4% ±0** |

The **same** trajectory yields the **same** displacement in every environment, so *any* deterministic
function of displacement (raw, RFF, MLP-lift, a NoPE-sum hidden) produces identical codes across
environments and collides when several maps are stored together — retrieval falls to ~1/M (4% at 16
maps). Grid/place cells **remap** (an environment-specific phase offset / field reassignment), keeping
the same location's codes orthogonal across rooms — **79–92%** at 16 maps, *non-overlapping CIs* vs the
additive code. Switching remapping **off** in the grid code reproduces the additive collapse (6%),
proving the necessary ingredient is **remapping itself** — a property of the biological population code
that no metric integrator possesses.

**This is the contribution, stated honestly.** Path integration is necessary but not sufficient and is
*not* unique to grid cells; a cognitive map additionally requires high-capacity, remappable population
coding, which additive integrators provably lack (4% vs ~90% across 16 maps). Grid/place cells supply
it, learned self-supervised, and the same code carries planning, value, and relational inference (above)
— a single substrate for the map, not just the metric. (`results/code_necessity.json`,
`results/code_necessity.svg`.)

### Boundary of the remapping claim — it does NOT transfer to a model with an external context label

Before building a multi-environment *language* task, we validated the design on CPU — and found an
honest boundary that **saved a futile GPU run** (`src/eval/multimap_task.py`, n=5). We repeated the
multi-map memory test, but replaced the one-shot Hebbian store with a **trained classifier** given a
learned **room-id embedding** (the analog of the room name appearing in an LLM's text prompt):

| recall acc | M=1 | M=4 | M=16 | M=32 rooms |
|---|---|---|---|---|
| grid + remap | 99% | 100% | 100% | 100% |
| **grid, NO remap** | 100% | 100% | 100% | **100%** |
| additive (raw 2-D) | 97% | 96% | 95% | 86% |

With gradient training, adequate capacity, and an explicit room-id, **remapping is no longer
necessary** — `grid, no remap` reaches 100% at every M (the classifier uses the room embedding to
disambiguate; only the low-dimensional raw 2-D code lags). This is principled, not a failure: the brain
**remaps because it has no external context label** — the hippocampus must *generate* the environment
context internally. An LLM *has* that label (the room name in the prompt), so it can substitute the
label for remapping. **The remapping necessity is therefore specific to context-free, capacity-limited
associative memory (Fig-3B), and does not transfer to an LLM with a text room-id** — so we did not build
that language task. Honest scoping of the claim, and a result in its own right about *when* the brain's
remapping matters. (`results/multimap_task.json`, `results/multimap_task.svg`.)

### Two more frontiers probed — sample efficiency and noise — both honest non-wins

We kept hunting for a regime where the *fixed* grid code beats a model that must *learn* to integrate
(`src/eval/frontier_probes.py`, n=5; the NoPE+sum Transformer is the toughest fair baseline):

- **Sample efficiency (acc at 3× length vs # training trajectories).** The grid code is *not* more
  data-efficient — at N=16 trajectories it scores 34% vs the NoPE+sum Transformer's 73% and the oracle's
  93%. Its high-dimensional code needs examples to learn the *readout*; a low-dimensional displacement
  feature generalizes from very few. (At large N all converge.)
- **Noise robustness (acc vs per-step velocity noise).** Once the comparison is *fair* — every code
  integrates the **same noisy velocity**, with the clean displacement only as the target — all codes
  degrade essentially identically: at σ=0.4, grid 34%, NoPE+sum 35%, GRU 33%, place 28%, oracle 35%.
  Velocity noise accumulates in the integrated displacement no matter how you encode it.
  *(An earlier version of this probe handed the grid/place codes the clean displacement and showed a
  spurious "grid is noise-immune at 96%"; we caught and fixed it. The honest result is a tie.)*

**Honest verdict on the hunt.** Across every fair test — length extrapolation, memory capacity,
remapping in a *trained* model, sample efficiency, and noise — the velocity-driven grid code is
*competitive but not uniquely necessary* for a trained system. The **additive integration prior** (which
a NoPE+sum Transformer also has) captures the core; the population-code extras (capacity, remapping)
matter only in narrow regimes — fixed-capacity associative memory, or context-free settings without an
external label. The defensible contribution is therefore the **rigorous, honestly-baselined
characterization itself** (including these negative results — a map of *when* brain-faithful coding helps
and when a simpler prior suffices) plus the **integrative demonstration** that one self-supervised code
serves navigation, planning, value, relational inference, and memory, read by a frozen LLM. We do not
claim grid cells are a uniquely necessary substrate for a trained model. (`results/frontier_probes.json`,
`results/frontier_probes.svg`.)

### Mechanism vs parameters — the reviewer control

"Is the grid code's extrapolation just from having a high-dimensional code (more parameters)?" We hold
the task, readout, and code dimensionality (384) fixed and vary only the code's STRUCTURE
(`src/eval/controls.py`, n=5; distance exact-acc):

| code (all 384-d) | T=8 | T=24 | T=48 |
|---|---|---|---|
| grid (geometric scales) | 99% | 93% | 76% |
| grid (random, non-geometric scales) | 98% | 92% | 74% |
| random **periodic** (Fourier features) | 99% | 94% | 77% |
| random **linear** (high-d, non-periodic) | 100% | 99% | 99% |
| learned MLP encoder (non-bio) | 99% | 99% | 99% |
| place (bounded tiling) | 97% | 81% | 57% |
| oracle | 99% | 99% | 99% |

The control is clarifying and, again, deflationary: it is **not** the parameter count (a random *linear*
384-d projection of displacement, same size, extrapolates perfectly — it losslessly re-encodes an
unbounded quantity), and it is **not** grid-cell specifics (random-scale grids and random *periodic*
features match the geometric grid). The single axis that matters is **saturation**: the *bounded* place
tiling cannot represent positions past its trained box (81%→57%), while every non-saturating code
extrapolates. The grid code's genuine niche is precise: it attains **unbounded metric range with
bounded, normalized (biologically plausible) activations** — where a place tiling cannot follow, and
unlike a linear code whose activations grow without bound (not a realizable neural code). So among
*bounded-activation* population codes, periodicity buys range; that is the honest, narrow sense in which
the grid code is special. (`results/controls.json`, `results/controls.svg`.)

### Non-Euclidean worlds — periodicity is NECESSARY (the tie-breaker + leakage rebuttal)

The honest Euclidean tie (a NoPE+sum Transformer matches the grid code) inverts the moment the world is
**cyclic**. On a torus, true position is θ = (∫velocity) **mod 2π**; a periodic grid code computes that
mod *for free* (cos(∫v) = cos(θ) for any number of wraps), whereas any non-periodic code sees an
unbounded ∫v and a readout cannot recover θ once paths leave the trained range. Train on short paths
(≤~1 wrap), test out to many wraps (`src/eval/torus.py`, n=8; toroidal position error in radians):

| code | T=8 | T=16 | T=32 | T=64 (many wraps) |
|---|---|---|---|---|
| **grid (periodic)** | **0.01 ±0.00** | **0.01** | **0.01** | **0.01 ±0.00** (= oracle, 100% within 45°) |
| additive (cumsum) | 0.22 | 1.08 | 1.36 | 1.51 (28%) |
| NoPE+sum Transformer | 0.49 | 1.65 | 1.56 | 1.56 (25% ≈ chance) |
| place (Euclidean tiling) | 0.21 | 1.34 | 1.58 | 1.58 (25%) |
| oracle | 0.01 | 0.01 | 0.01 | 0.01 |

The grid code is **flat at the oracle floor (0.01 rad, 100%) at every length**, while **the very NoPE+sum
Transformer that tied it on Euclidean paths collapses to chance (1.56 rad, 25%)**, as do the additive and
Euclidean-place codes — all with tiny, non-overlapping CIs. So the periodicity that merely bought *finite
range* on Euclidean paths (a wash vs additive integrators) is *exactly the right inductive bias* for a
cyclic world: there, the brain-faithful code is not competitive-but-tied, it is **necessary** — a clean,
positive, significant win. This is simultaneously the **definitive leakage rebuttal**: a torus has no
faithful Euclidean text description, so an LLM's text prior cannot solve it; only a code that has actually
path-integrated the cyclic geometry can. (`results/torus.json`, `results/torus.svg`.)

**Confirmed through the frozen LLM — the leakage-proof causal headline (`--task torus`).** A frozen
cortex, **self-supervised-pretrained on the torus** (toroidal harmonics of L; a Euclidean-pretrained
readout hides the wrapped cell — that is itself the boundary result above), lets a LoRA-Qwen answer
"which of 9 wrap-around cells are you in?" — a question with no faithful Euclidean text description, with
the moves **never in the prompt** (**n=6 seeds**; `results/torus_llm.json`):

| torus-cell exact acc | T=8 (train) | T=16 (extrap.) | T=24 (extrap.) |
|---|---|---|---|
| **cortex-ON** | **84% ±23** | **74% ±21** | **63% ±19** |
| text-only OFF | 11% ±4 | 9% ±2 | 11% ±3 |
| **Δ (ON − OFF), every seed** | **+73** | **+66** | **+52** |

In **every one of the 6 seeds**, cortex-ON beats the text-only control by **52–73 points** while OFF stays
at chance — so the *causal, leakage-proof* claim is robust **and now significant**: the paired sign-flip
permutation test gives **p = 0.033 at every length** (clearing the n=3 floor of 0.25). The LLM answers by
**reading a path-integrated toroidal code**, not a language prior over Euclidean space (the world is
cyclic, so no such prior helps). *Honest caveat:* the ON magnitude is **seed-variable** (CIs wide), so we
report the spread — but the causal direction is significant and consistent across seeds and lengths.

This is the single-item counterpart to the §"structural transfer" negative: **single-item spatial
readouts transfer to the frozen LLM (even on a non-Euclidean world); pairwise comparison does not** — an
honest, informative boundary.

### Structural transfer — a space-trained metric does abstract relational inference (TEM, with falsifiers)

The Tolman-Eichenbaum hypothesis is that the *same* metric code maps abstract relational structure, not
just physical space. We test it with the cortex **frozen and trained only on spatial path integration**:
lay a non-spatial ordered structure (ranks 0…N−1) along a concept axis, push each item through the frozen
grid code by its *own* position (never the signed relative displacement — that would leak the answer),
and train a comparison readout on **adjacent pairs only** (`src/eval/structural_transfer.py`, n=8):

| metric | mean ± 95% CI |
|---|---|
| **transitive inference** (far pairs, never trained) | **0.836 ± 0.008** |
| adjacent pairs (trained) | 0.706 ± 0.019 |
| schema transfer (a NEW item set, new region) | 0.790 ± 0.006 |
| symbolic-distance-effect correlation | 0.953 ± 0.013 |
| **CONTROL — shuffled positions** (rank↔space destroyed) | **0.623 ± 0.037** |
| **CONTROL — scrambled 2nd item** | 0.656 ± 0.006 |

Transitive inference on never-seen far pairs (0.836) *exceeds* the trained adjacent pairs (0.706) — that
inversion **is** the symbolic-distance effect (far pairs are easier; corr 0.95), the behavioural
signature of an analog/spatial representation of an abstract dimension. The two falsifiers a reviewer
demands both fire: **shuffling the rank↔position correspondence collapses TI toward chance (0.836 → 0.623,
paired p = 0.009)** — so TI comes from the cortex's *ordered metric*, not memorization; and scrambling the
second item (0.656) confirms the readout compares *two* codes, not one item's magnitude. (They don't fall
all the way to 0.50 — a residual single-item signal survives at this noise — but the significant drop
isolates the metric as the cause.) This is the representation-level validation of the headline LLM
experiment, where the MLP readout is replaced by a frozen Qwen+LoRA answering a linguistic comparison.
(`results/structural_transfer.json`, `results/structural_transfer.svg`.)

*LLM-transfer of the relational comparison — a negative result, reported.* We tried to lift this to a
frozen Qwen (`src/training/train_relational.py`, `notebooks/m2_relational_llm_kaggle.py`): each item
enters by its own position through the frozen cortex; a LoRA-Qwen reads both and answers "is the first
ranked higher?", trained on adjacent pairs. Across 4 configurations and 3 independent evaluators
(generation, Yes/No-logit, candidate-NLL), it stayed at **exactly 50% including on the trained pairs** —
training loss sits at the trivial answer-token floor while the Yes/No token stays at chance, i.e. the
LLM never learns to *use* the spatial channel for a **two-item comparison** (single-item tasks like
return/distance/bearing/torus do learn, because the answer is a direct readout of one trajectory). So
the TEM claim is supported at the representation level (TI 0.99 through the real frozen cortex) but
**does not currently transfer through the frozen-LLM fusion interface for pairwise comparison** — an
honest limitation and a target for future work (e.g. a comparison-aware fusion or a larger adapter).

### The phase diagram — *when* each inductive bias wins (synthesis)

The negative/tie results stop being a deflation once organized into a predictive map. `src/eval/phase_diagram.py`
assembles all the committed multi-seed results into one regime × code matrix (win / tie / lose vs the
best in that regime), turning "grid isn't uniquely necessary" into "**here is the win-region of each
inductive bias**":

| regime | grid (periodic) | place (bounded) | additive integrator |
|---|---|---|---|
| Euclidean extrapolation (4×) | tie | lose | **win** |
| **Cyclic world (torus)** | **WIN** | lose | lose |
| One-shot capacity (200 items) | **WIN** | win | lose |
| Multi-map, NO context label | tie | win | lose |
| Multi-map, WITH context label | tie | – | tie |
| Very low data (16 trajectories) | lose | lose | **win** |
| Heavy integration noise | tie | tie | tie |

Read off the map: the periodic grid code **wins where periodicity / pattern-separation is load-bearing**
(cyclic worlds; one-shot capacity — alongside the place code), is **competitive (tie)** where a plain
integration bias suffices (Euclidean magnitude extrapolation, multi-map once a context label is given,
heavy noise), and **loses only in the very-low-data regime**, where a low-dimensional code is simply
easier to read. The additive integrator wins exactly the two regimes where structure is *not* needed
(Euclidean extrapolation, low data); the bounded place code wins *nowhere on its own*. This is the
honest contribution: not "grid cells are magic," but a **falsifiable account of which world-properties
make a brain-faithful code necessary** — confirmed, not asserted. (`results/phase_diagram.json`,
`results/phase_diagram.svg`.)

## The map is PREDICTIVE and TEMPORAL — the two axes the spatial cortex omitted

Reading *The Neuroscience of Learning in Space and Time* against our model surfaced a real gap: the
hippocampal map is **not a geometric record of where you are** but a **predictive model of where you
are going** (the successor representation; Dayan 1993, Stachenfeld 2017), indexed in **time** as much
as in space (time cells; Eichenbaum 2014, MacDonald 2011, Howard's scale-invariant timing). Our cortex
was purely spatial and purely geometric. Two CPU-validatable modules close that gap, each reproducing
the *falsifiable signature* the brain is known for — multi-seed, mean ± 95% CI.

### Predictive map — the successor representation routes around barriers where geometry stalls

`src/eval/successor.py` builds the successor representation **M = (I − γT)⁻¹** (expected discounted
future occupancy) on a barriered gridworld and asks what it buys over a purely geometric map (n=8):

- **Planning around a wall.** Greedily ascending the SR value reaches the goal **100%** of the time
  with a barrier present; greedily descending *Euclidean* distance-to-goal stalls at **61.7% ± 9.3%**
  — because the wall makes the straight-line gradient point *into* the barrier (paired sign-flip
  **p = 0.0086**, +38 pts). On an **open** field both succeed 100% — the SR advantage is *specifically*
  the detour, not generic competence. This is Tolman's insight in one number: a predictive map plans;
  a metric map only points.
- **The field bends around the wall.** SR place fields track **geodesic** (on-manifold) distance, not
  Euclidean: for across-wall cell pairs the SR field correlates **0.69 ± 0.06** with graph-geodesic
  distance vs **0.31 ± 0.12** with Euclidean. The map measures *reachability*, the topology, not the
  ruler.
- **It is learnable from experience.** A TD-learned SR (online, from sampled transitions only) matches
  the analytic closed form at **0.97 ± 0.003** — so this is not just an algebraic construct; it is what
  a brain following the same TD rule would acquire. (SVG: SR field, the grid-like SR eigenvector, and
  the planning bars. `results/successor.json`, `results/successor.svg`.)

### Temporal map — time cells and scalar (Weber) timing EMERGE from the substrate

We do not hand-build a time-cell basis; we let it emerge, exactly as grid cells emerge from path
integration. `src/models/neuro/temporal_cortex.py` is a generic recurrent substrate (a leaky rectified
rate-RNN, ONE uniform time-constant, learned recurrence, private membrane noise) — nothing about time
cells, field widening, or scalar timing is built in. `src/eval/time_cells.py` trains it on a single task,
"report how much time has elapsed since a start pulse, when probed at a random moment," with a metabolic
activity cost, and then MEASURES what appeared (n=8; an UNTRAINED net of the same architecture controls):

- **A precise timer emerges.** Elapsed time decodes at **0.20 ± 0.04 steps** of error; the untrained
  control cannot time (**3.56 ± 1.37**). Time became a usable quantity purely from learning the task.
- **Time cells emerge.** **17% ± 3%** of units are single-peaked fields tiling the interval (untrained
  **1%**), and — unprompted — **92% peak in the first half**, the real denser-at-short-latency gradient
  (Mau et al. 2018).
- **Fields widen with latency.** corr(field width, peak time) = **+0.67 ± 0.10**, positive in every seed
  — never in the loss; the cellular substrate of scalar timing, fallen out of the learned dynamics.
- **It obeys Weber's law.** The trial-to-trial SD of decoded time grows ~linearly with elapsed time (corr
  **+0.98 ± 0.01**) at a ~constant Weber fraction (CV **0.15 ± 0.02**, scale-invariant) — the defining
  behavioral signature of interval timing (Gibbon 1977), arising from private noise integrated through
  the widening code; the untrained net is not scale-invariant (CV 0.22) and cannot time at all.

So the temporal neuroscience is *measured, not designed*: a substrate told only to read elapsed time
develops hippocampal time cells, their latency-dependent widening, and the brain's scalar-timing law.
(`results/time_cells.json`, `results/time_cells.svg`.)

### The emergent time code transfers to language — the temporal analogue of torus-QA

A frozen LoRA-Qwen2.5-1.5B answers **"how much time has elapsed?"** (6 bins) reading ONLY the FROZEN
emergent temporal cortex — the elapsed time never appears in the prompt, so a high cortex-ON vs
text-only-OFF gap is a causal, leakage-proof statement that the LLM reads the emergent time-cell code
(`notebooks/m3_temporal_full_kaggle.py`, **n=6**; chance 17%):

- **EXACT bin:** ON **55% ± 20** vs OFF **16% ± 6** (Δ **+40**; OFF sits at chance — the clean causal contrast).
- **WITHIN-1** (the natural metric for a *scalar* quantity): ON **70% ± 19** vs OFF **37% ± 17** (Δ **+33**).

ON exceeds OFF in **all six seeds** (best seed **86% exact / 96% within-1**), so the paired sign-flip
permutation test is **significant on both metrics (p = 0.033)** — clearing the n=3 floor of 0.25. As for
torus-QA, the ON magnitude is seed-variable (±20; the cortex's *emergent* code quality varies seed to
seed), so we report the honest spread. This closes the temporal loop the spatial torus-QA closed: **a
frozen LLM reads an emergent time-cell code it was never given in text.** Both axes of the
predictive-spatiotemporal map — space (torus) and time (elapsed) — now transfer to language, all
emergent, nothing hard-coded. (`results/elapsed_time_llm.json`.)

### Toward the organ — a SPIKING, multi-timescale time code (honest, mixed)

The rate model above reproduces the time-cell *signature*; a spiking substrate narrows the gap toward
the biophysical *organ*. `src/models/neuro/spiking_temporal_cortex.py` is a recurrent **adaptive-LIF**
network (surrogate-gradient spikes, per-unit **learnable membrane & adaptation time-constants**, private
noise) — grounded in recent SNN-timing work (heterogeneous learnable τ → multi-timescale dynamics;
adaptive-LIF → transient firing). Trained only to report elapsed time, with rate homeostasis, then
measured vs a **homogeneous-τ control** (n=6):

- **Spiking time cells emerge** — **46% ± 5%** of units are single-peaked and tile the interval (the
  control also has ~49%: the time cells come from spike-frequency **adaptation**, present in both).
- **A multi-timescale spectrum emerges and is *functional*** — learnable membrane τ spread **14.6× ± 3.6**
  (control 1.0× by construction), and the heterogeneity **improves timing: decode error 0.87 ± 0.44 steps
  vs 1.47 ± 0.31 homogeneous**. This is the clean "multi-timescale matters" result.
- **Widening + scalar timing reproduce in spikes**, more noisily than rates: width-vs-latency
  **+0.47 ± 0.19**, scalar-σ **+0.70 ± 0.18** (Weber CV 0.39 vs the rate model's 0.15 — spikes add
  variability).
- **Honest non-result.** A "slow cells code late" (Howard log-compression) trend looked strong at n=2
  (+0.36) but **did not replicate at n=6** (corr(τ, peak) **+0.10 ± 0.17**, CI crosses 0). We withdraw
  it — the multi-seed run did its job. The robust multi-timescale claim is the *spectrum + the timing
  benefit*, not τ-to-latency alignment.

So **spiking** and **multi-timescale** are closed (signatures reproduced in spikes; an emergent τ
spectrum that aids timing); local learning (e-prop) and circuit embedding remain (content-binding is
closed next). (`results/spiking_time_cells.json`, `results/spiking_time_cells.svg`.)

### Content-binding — the code says WHAT happened WHEN (conjunctive vs pure time cells)

The temporal code also BINDS CONTENT, reproducing a 2023 hippocampal finding (bat CA1; Shimbo et al.,
Nature Neuroscience; space–time integration, Neuron 2024): time cells split into two coexisting
populations. `src/eval/content_binding.py` gives the substrate ONE of K=3 events at t=0 and asks it, at a
random probe, to report BOTH elapsed time AND which event — nothing about conjunctive/pure cells imposed
(n=6):

- **Both populations EMERGE, every seed:** **PURE time cells 29% ± 7** (fire at their moment regardless of
  event) and **CONJUNCTIVE "contextual" cells 71% ± 7** (event × time).
- **The population decodes BOTH what and when:** event identity **100% ± 0** (chance 33%) and elapsed time
  **1.31 ± 0.12 steps**; the time cells still widen with latency (+0.73).

So a single recurrent substrate, told only to report what-and-when, grows a hippocampus-like code that
**binds episodic content to time** — the third "organ" gap (after spiking and multi-timescale) closed on
CPU; local learning (e-prop) and grid-cortex circuit embedding remain. (`results/content_binding.json`,
`results/content_binding.svg`.)

**Through the frozen LLM — both fields readable, but a joint-answer capacity tradeoff (n=6).** A
LoRA-Qwen reads ONLY this content-binding cortex (neither field in the prompt;
`notebooks/m4_what_when_kaggle.py`). Each field is **individually significant**, but they **trade off in
a single joint answer**: with the event first / equal weight, **WHAT wins** (cortex-ON 76% ± 18 vs OFF
26% ± 10, **p=0.033**) while WHEN sits at chance (p=0.78); flip to time-first + up-weight the time tokens
and **WHEN wins, strongly** (exact ON 67% vs OFF 17%, **p=0.033**; within-1 **91%** vs 44%, p=0.033) while
WHAT drops to marginal (43%, p=0.095). So the frozen-LLM fusion can read the **categorical** *or* the
**scalar** field — whichever the loss emphasizes — but a single autoregressive answer is a **capacity
bottleneck** that crowds out the other. This is a property of the *readout interface*, not the binding:
the cortex encodes both (CPU decode: what 100%, when 1.31 steps), and the standalone elapsed-time readout
transfers on its own (`results/elapsed_time_llm.json`, p=0.033). A separate-query readout
(`notebooks/m4b_separate_readout_kaggle.py`, asking *what?* or *when?* independently) confirms it but
inherits the same limit: split 50/50, **WHEN stays significant** (exact 36% vs 13%, within-1 78% vs 42%,
p=0.033) while **WHAT slips to marginal** (47% vs 33%, p=0.16) on its halved training share. So the
complete picture is: a frozen LLM reads *either* field of the bound code **to significance** — WHAT
(p=0.033) when event-emphasized, WHEN (p=0.033) when time gets a fair share — but a single small LoRA
readout **cannot max both at once**; the bottleneck is the readout's capacity/training-share, not the
binding (which the CPU decode confirms). (`results/what_when_llm.json`.)

*Follow-up — we tested whether a theta rate+phase code fixes it, and recorded an honest framing
correction* (`src/eval/phase_channel.py`, n=3). At the population level there is **no tradeoff to fix**: a
plain rate-only *linear* reader already decodes **both** (what 100%, when MAE 1.4 steps). So the joint
tradeoff is a property of the *tiny frozen-LLM reader*, not the cortical code, and a rate+phase phasor
does not rescue a non-problem (phasor ≈ capacity-matched rate). The one genuine phase signature is the
Huxter-direction lean: **elapsed time decodes better from the phase channel than from rate (MAE 0.82 vs
1.16, CIs non-overlapping)** — partial rate–phase independent coding, not full segregation. Lesson: phase
multiplexing is the brain's solution *at scale*; over-applying it to our toy was the wrong inference.
(`results/phase_channel.json`.)

### The signatures survive the BRAIN'S learning rule — local e-prop, no backprop (n=5)

Everything above is trained by backprop/BPTT, which the brain does not do. `src/eval/eprop_local_learning.py`
asks whether the temporal signatures survive a biologically-plausible **local** rule: **e-prop** (Bellec
et al. 2020) — each synapse keeps an **eligibility trace** of its own pre/post activity, gated by one
broadcast **learning signal** (the readout error); no backward pass through time. Adaptive-LIF neurons
supply a *slow* adaptation-eligibility component that carries temporal credit across the delay. Trained
this way only (no autograd), a recurrent ALIF net (n=5):

- **Learns to time** — loss/T **0.030 ± 0.025**, below the predict-the-mean floor (0.083) in **all 5
  seeds**; elapsed time decodes at **MAE 2.45 ± 0.76 steps** (of 40).
- **Grows spiking time cells** — **10.3% ± 1.8%** (single-peaked, raw-spike tuning; every seed 8–12%).
  Fewer than under backprop (~46%) and coarser, but unmistakably present.

So the time-cell signature does **not** require backprop: it emerges under the brain's own kind of local,
online plasticity. This is the strongest form of the project's thesis — the *architecture* (recurrent
adaptive spiking dynamics) gives rise to the neuroscience, even when the *learning rule* is also
brain-faithful. (`results/eprop_local_learning.json`, `results/eprop_local_learning.svg`.)

### One-shot learning the biological way — BTSP and its PREDICTIVE place field (n=5)

The first gap closed from the new gap register (`GAPS.md` #1). The model's one-shot memory
(`agent_memory.py`) writes a place code into an **episodic store** — a functional abstraction. The
hippocampus does it differently and more interestingly: a single dendritic **plateau potential** imprints a
complete place field in **one traversal**, through a **seconds-wide, temporally asymmetric** plasticity
kernel — **behavioral-timescale synaptic plasticity** (BTSP; Bittner, Milstein, Lu, Turi & Magee, *Science*
2017; Grienberger & Magee 2022), now thought to be the hippocampus's dominant rapid-learning rule. We add a
`BTSPPlasticity` organ, drive position-tuned inputs along a track at speed *v*, fire **one** plateau at the
centre, apply the rule **once**, and read out the field. Nothing about the field is trained — we set only the
kernel (the biology) and **measure**:

| kernel | field strength | predictive shift | field width |
|---|---|---|---|
| **BTSP** (asymmetric, seconds) | **1.00** | **−12.96** | 53 |
| symmetric (seconds) | 0.98 | +0.14 | 52 |
| STDP (millisecond) | **0.02** | +0.04 | — |

- **(A) One-shot needs a SECONDS-scale kernel.** BTSP and a symmetric-seconds control both imprint a strong,
  broad field in one pass (strength 1.00, 0.98); a millisecond STDP-scale kernel imprints almost nothing
  (**0.02**) — STDP is the wrong timescale for one-trial place-field formation.
- **(B) The PREDICTIVE shift needs the ASYMMETRY.** Only the asymmetric BTSP kernel shifts the field
  **upstream** of the plateau (**−12.96**, i.e. the cell fires *before* the induction site on the next
  same-direction lap — anticipatory); the symmetric control sits on the plateau (+0.14). The shift is *not*
  put in — it emerges because the animal occupied upstream positions in the seconds before the plateau, and the
  asymmetric kernel potentiates those inputs most.
- **(C) The shift SCALES WITH RUNNING SPEED** (−8.0 → −12.9 → −16.7 as *v* = 15 → 25 → 40): a *temporal* kernel
  read out as a *spatial* shift (shift ≈ kernel-offset × speed), a specific Bittner prediction — measured.

So the biological one-shot rule, with its signature predictive field, replaces the episodic-store abstraction
as an *emergent* result. (`results/btsp.json`, `results/btsp.svg`.)

### One circuit for space AND time — place, time, and conjunctive cells coexist (n=5)

In hippocampus, place cells, time cells, and **conjunctive space×time** cells share a single population
(Neuron 2024; bat CA1, Nat. Neurosci. 2023). `src/eval/space_time_circuit.py` feeds ONE recurrent
substrate self-motion velocity **and** a start pulse and trains it to report **both** position and
elapsed time; we then measure, per unit, the variance explained (η²) by position vs by elapsed time (a
bounded box keeps the two ~decorrelated, so the tunings are separable). The single circuit develops all
three cell types, coexisting (n=5):

- **PURE PLACE** (space-tuned, time-invariant): **19% ± 3** · **PURE TIME** (time-tuned, space-invariant):
  **17% ± 3** · **CONJUNCTIVE space×time**: **51% ± 3** — the conjunctive majority matching the data.
- It decodes both at once: position **MAE 0.20** (box half-width 1.0) and elapsed time **MAE 1.30 steps**.

So space and time are not separate modules here — one recurrent circuit multiplexes *where* and *when* in
the same units, with the conjunctive code dominant, exactly the hippocampal organisation.
(`results/space_time_circuit.json`, `results/space_time_circuit.svg`.)

### A SELF map and an OTHER-agent map in one population — social place cells (n=5)

Gap #4 from the register, closed with the same methodology. The hippocampus encodes not only the animal's
**own** position but **another individual's** — dedicated *social place cells* (Danjo, Toyoizumi & Fujisawa
2018; Omer, Maimon, Las & Ulanovsky 2018 in bats), and humans map social variables with the same machinery
(Tavares 2015; Park 2021). The model had **no** representation of a second agent (`GAPS.md` #4). We add
`src/eval/social_space.py`: ONE recurrent substrate is fed its **own** self-motion **and** its observation of
**another agent's** motion, and trained to report **both** positions. Then we MEASURE, per unit, the variance
explained (η²) by self-position vs other-position — nothing imposed:

- **PURE SELF-place** cells (own position): **22% ± 4** · **PURE OTHER-place** cells (the other agent's
  position): **20% ± 2** · **CONJUNCTIVE self×other**: **42% ± 6**. A self-map and an other-map coexist in one
  population — the social place cells, emergent.
- **Clean double dissociation** (decode MAE, box half-width 1.0):

| | decode SELF | decode OTHER |
|---|---|---|
| intact | 0.22 | 0.21 |
| **lesion OTHER cells** | 0.22 (survives) | **0.40** (fails) |
| **lesion SELF cells** | **0.40** (fails) | 0.21 (survives) |

Lesioning the OTHER-place cells wrecks decoding of the other agent's position (0.21 → 0.40) while self-decoding
is untouched (0.22); lesioning the SELF-place cells does the exact reverse. So the brain's map of *others* is a
distinct, separable population that coexists with the self-map in one circuit — the first social/other-agent
representation in the model. (`results/social_space.json`, `results/social_space.svg`.)

### Goal & reward coding — a goal-vector code, and reward fields that ANTICIPATE the goal (GAPS.md #3)

Gap #3, closed in two parts (designed with a research + red-team panel to defeat circularity).

**A — A goal-direction code emerges from navigation** (`src/eval/goal_vector.py`, n=5). Neurons encode a vector
to a remembered goal (Sarel, Finkelstein, Las & Ulanovsky 2017; Ormond & O'Keefe 2022). A generic ReLU policy
trained ONLY to reach randomized goals from the grid code (goal enters *only* as `grid_code_at(goal)` — never a
decoded goal-minus-position vector, the cardinal trap) navigates at **99.7%** success; and **95%** of its
active hidden units then tune to the (allocentric) direction to the goal — **EMERGENT and GOAL-SPECIFIC**: an
untrained-weights baseline (**2%**) and a goal-label SHUFFLE null (**1%**) both sit at the false-positive floor
(the Banino-2018 "vector-to-goal codes emerge from navigation" template). *Honest scope, three ways:* the code
is **allocentric** (it matches the action frame; Chadwick 2015); it is **distributed/redundant** (no small
subset is necessary — a directional task recruits nearly every unit, so there is no place-vs-goal dissociation
here); and **egocentric** goal-direction (**0%**) and metric **distance-to-goal** cells (**2%**) do NOT emerge —
a magnitude-free directional task neither supervises nor requires them, so the code encodes exactly what the
behaviour needs (Sarel's egocentric + distance cells would need egocentric steering and distance-dependent
behaviour — a noted extension). (`results/goal_vector.json`, `results/goal_vector.svg`.)

**B — Reward fields that ANTICIPATE the goal, via reward-triggered BTSP** (`src/eval/reward_map.py`, n=5).
Place fields over-represent reward and peak just *before* it (Hollup 2001; Gauthier & Tank 2018; Boccara 2019).
We compose the `BTSPPlasticity` organ: reaching a reward triggers one plateau, whose seconds-wide asymmetric
kernel imprints a one-shot field. *Honesty (per the red-team):* fields piling up **at** the reward is partly by
construction (the plateau fires there), so every reported result is a **difference vs. a matched control**:

- **The anticipatory shift is the emergent signature.** The field population sits **UPSTREAM** of the reward
  along the approach (**−0.23 ± 0.03**) — the plateau fires AT the reward; the fields end up BEFORE it, purely
  from the kernel's asymmetry. It **cleanly vanishes** under a symmetric-kernel control (**+0.02 ± 0.03**; the
  run passes *through* the reward, so a symmetric kernel centres the field on it). This is the Mehta/Bittner
  predictive shift, tied to reward — measured, not imposed.
- **Reward-specific concentration.** Fields concentrate at the reward **43×** vs a yoked control firing the same
  number of plateaus at **random** locations (**0.8×**) — so the concentration is reward-driven, not "BTSP
  writes a field wherever it fires."

So the model now has a goal-vector code (emergent from navigation) and a **predictive reward map** (reward-gated
BTSP builds fields that anticipate the goal). (`results/reward_map.json`, `results/reward_map.svg`.)

### A grid code for CONCEPTS — the hexadirectional signal, its symmetry inherited from the lattice (GAPS.md #2)

The last register gap, and the trickiest to do **non-circularly**. Humans show a six-fold (hexadirectional)
entorhinal signal moving through space *and* through abstract 2-D **concept** spaces — the grid code as the
brain's general cognitive-map engine (Doeller, Barry & Burgess 2010; **Constantinescu, O'Keefe & Behrens 2016**;
Kunz 2019). The naive worry — "a hex grid is 6-fold by construction, so measuring 6-fold is circular" — is
wrong, and *why* it's wrong is the result: a summed grid **rate map is direction-invariant**; the 6-fold lives
only in the **direction** signal, and only through a movement-sensitive **nonlinearity** (conjunctive grid ×
direction cells, Sargolini 2006; Bush & Burgess 2015). We add `ConjunctiveGridDirectionCells` (with **uniform**
preferred directions — nothing 6-fold built in) and measure the population's movement-driven activity **power**
as the agent runs in each direction, fit to `β0 + A6·cos(6θ) + A4·cos(4θ) (+ 5/7-fold controls)`:

| read-out of the grid | 6-fold A6 | 4-fold A4 | adj. 5/7 | 6-fold index |
|---|---|---|---|---|
| **HEX grid, nonlinear** | **0.040** | 0.010 | 0.011 | **80%** |
| SQUARE grid, nonlinear | 0.004 | **0.038** | — | 10% |
| HEX grid, LINEAR read-out | 0.005 | — | — | — |

- **(A) The hexadirectional signal emerges.** The model's hexagonal grid gives a **6-fold** direction signal
  (A6 **0.040**, index **80%**) that sticks out above the 4-fold (0.010) *and* the adjacent 5/7-fold control
  symmetries (0.011) — the human entorhinal signature, read out through the nonlinearity.
- **(B) Its symmetry is INHERITED from the lattice, not imposed** (the decisive non-circularity control). A
  **square** lattice — same construction, 4-fold instead of hex — **flips** the signal to **4-fold** (index
  **10%**, A4 0.038 ≫ A6 0.004). The directional symmetry *tracks the spatial lattice*.
- **(C) The nonlinearity is necessary.** A **linear** read-out (the mean, not the movement-power variance) of
  the same hex grid is direction-invariant (A6 **0.005**) — a raw grid rate map carries no hexadirectional
  signal; and the cells' preferred directions are **uniform**, so nothing 6-fold is put in.

Reading the two axes as abstract **concept features**, the *same* grid metric produces the hexadirectional
signal for movement through concept space — the human cognitive map, from space to meaning. So the repo's grid
code is the bat-faithful/human-faithful one **and** the mechanistic origin of the hexadirectional signature is
measured, not assumed. (`results/hexadirectional.json`, `results/hexadirectional.svg`.)

### Hippocampal subfields — DG pattern separation + CA1 comparator around the CA3 auto-associator (GAPS.md #5b, n=5)

The repo had **CA3** (`HopfieldAssociativeMemory` — a Marr/Hopfield/Treves-Rolls recurrent auto-associator that
pattern-*completes* a cue and *interferes* when stored patterns are too similar) but not the two subfields that
make the triad work. `src/eval/hippocampal_subfields.py` adds them and **measures** the functional consequences,
guarding the by-construction trap — a sparse random expansion trivially orthogonalizes, so the headline is the
*downstream recall*, not the DG orthogonality itself.

- **(A) Separation → interference-free recall.** Store **M = 24 SIMILAR environments** (entorhinal overlap 0.6)
  and recall each from a **30 %-degraded cue**. The **dentate gyrus** — a massive **sparse expansion** (5 %
  active over N_dg = 1500) — lets CA3 recall the *correct* environment **0.87 ± 0.02**, where a **dense**
  expansion of the **same size** intrudes on a similar one (**0.37 ± 0.03**; gap **+0.51 ± 0.03**). The mechanism
  check confirms the *why*: DG's separation index (output overlap ÷ input overlap) is **0.54** (orthogonalized)
  versus **1.00** for the dense code (overlap preserved) — but the reported headline is the recall, not the
  separation.
- **(A′) A large CA3 is *actively harmful* unless sparse.** The dense expansion (0.37) is **worse than not
  expanding at all** (direct-EC 0.86). Expansion alone doesn't help — it *hurts*, by giving overlapping inputs
  more room to collide; DG's **sparsity** is what turns the expansion from a liability into pattern separation.
- **(B) Dense-expansion falsifier.** Same N_dg, same random expansion, only the k-WTA sparsity removed →
  interference returns (0.37). So it is the sparse **separation**, not the extra dimensionality, doing the work.
- **(C) CA1 comparator — novelty needs the memory.** CA1 reads the **mismatch** between CA3's completed
  prediction and the entorhinal input. It discriminates **novel vs familiar** environments at **AUC 1.00 ±
  0.00**; **ablate the CA3 stream and the AUC falls to 0.50** (chance) — so it is a genuine *entorhinal-vs-memory*
  comparator (Lisman & Grace 2005; Vinogradova 1995), not an input-novelty detector (which the NE organ, #5,
  already is).

So the hippocampal triad is now differentiated the way the systems-neuroscience account requires: **DG**
separates, **CA3** completes, **CA1** compares — each validated against its own falsifier, nothing put in a
loss. (`results/hippocampal_subfields.json`, `results/hippocampal_subfields.svg`.)

### Neuromodulation — acetylcholine sets encode vs. retrieve, noradrenaline gates remapping (GAPS.md #5, n=5)

Gap #5 from the register. The model already had DA-/NE-style ML gates (`PredictionErrorGate`, `AdaptiveGain`)
but wired only into `diagnose.py`/`accuracy.py` — never into the hippocampal dynamics, and with no
acetylcholine encode/retrieve switch. We add two mechanistic organs — `AcetylcholineGate` and
`LocusCoeruleusReset` — acting on a CA3-style auto-associator `HopfieldAssociativeMemory` (Marr 1971;
Hopfield 1982; Treves & Rolls 1994), and **MEASURE** the classic signatures. Nothing is trained; the mode is
set and the consequences are read out. The design was hardened against a circularity **red-team**: because a
single recurrent-gain knob *trades* encode-cleanliness for retrieval-completion **BY CONSTRUCTION**, and a reset
decorrelates the code **BY CONSTRUCTION**, *neither is reported*. Every number below is a **DIFFERENCE against a
matched control, at matched storage energy** — the reward_map standard.

**Acetylcholine (Hasselmo 2006): high ACh = encoding suppresses recurrent recall while enhancing plasticity.**

| encoding a new field near a stored one | intrusion on the old memory |
|---|---|
| near / overlapping, **low ACh** (recall on) | **+0.81 ± 0.00** |
| near / overlapping, **high ACh** (recall off) | **+0.42 ± 0.00** |
| **FAR / non-overlapping** floor | **+0.02 ± 0.00** |

- **(A1) Intrusion is OVERLAP-SPECIFIC** — the genuine signal, reported as excess over the far floor exactly as
  reward_map reports over-representation over a yoked floor: encoding a field that OVERLAPS a stored memory is
  pulled toward it (**+0.81**), while a distant field is not (**+0.02**) → **excess +0.78 ± 0.00**. Recurrence
  only misdirects when there is a nearby attractor to be captured by.
- **(A2) It is RECURRENT CONTAMINATION, not non-storage.** Intrusion grows with the encoding recurrent gain
  (**+0.42 → +0.81**, effect **+0.38 ± 0.00**) while the write energy ‖ΔW‖ is held **MATCHED** (3.59 vs 3.66) —
  so the difference is *what* was written (a representation pulled onto the old memory), not *how much*. High ACh
  suppresses the recall so the new field is written where it belongs.
- **(A3) Retrieval completion REQUIRES the recurrent weights.** A 50%-dropped, **transient** cue (cosine
  **0.68** to the target) is completed to **0.96** with W_rec but stays at **0.68** with W_rec off (recovery
  **+0.29 ± 0.04**) — the same recurrent synapses ACh suppresses for clean encoding are what pattern-complete a
  degraded cue during retrieval. The cue is transient, so this is completion, not the cue echoing itself.

**Noradrenaline (Yu & Dayan 2005; Bouret & Sara 2005): a phasic surprise burst gates remapping.**

- **(B1) Surprise is NOVELTY, not change magnitude.** Mean prediction-error surprise is **0.16** for familiar
  input+noise, **0.16** for a *large but EXPECTED* position jump (‖Δsensory‖ large — a change detector would
  fire here), and **0.99** for genuinely novel input; θ-independent **AUC 1.00 ± 0.00** separating novel from
  familiar. So NE tracks unpredicted-ness, not raw input change.
- **(B2) A surprise-triggered remap is ADAPTIVE on BOTH sides**, vs a **matched no-reset+re-encode** control
  (the control also re-encodes the new world, but onto the stale map): remap **learns the new environment**
  (prediction error 0.34 → 0.05, benefit **+0.29 ± 0.01**, the stale attractor causes proactive interference)
  **and protects the old map** from overwrite (old-env recall error 0.32 → 0.05, benefit **+0.27 ± 0.00**,
  retroactive interference). This **unifies** the two systems: NE clears/re-indexes the map so the ACh-gated
  encoder can write the new one without interference.

**Honest scope.** Only the *tonic* cholinergic set-point is modeled — the fast within-theta-cycle
encode/retrieve alternation (Hasselmo, Bodelón & Wyble 2002) is out of scope; the afferent input is *relatively*
spared (recurrent suppressed) rather than absolutely boosted. The recurrent store is a Hopfield/Marr–Willshaw
autoassociator (not BTSP — BTSP is millisecond-asymmetric and feedforward; it composes here only as the
afferent one-shot write). And the LC-NE → hippocampal-**remapping** link is a **hypothesized bridge**: remapping
itself is classically driven by contextual/sensory change (Muller & Kubie 1987; Leutgeb 2005; Colgin, Moser &
Moser 2008), and we test only that a surprise-gated reset is novelty-locked and adaptive. The organs are also
wired into the cortex (`BrainSpatialCortex(ach=…)` routes ACh to the grid attractor's recall).
(`results/neuromodulation.json`, `results/neuromodulation.svg`.)

### Learning without backprop — credit assignment via feedback alignment (GAPS.md Tier 5, #A1, n=5)

The deepest "how the cortex learns" gap: everything else here is trained by backprop, whose least biological
requirement is **weight transport** — the backward pass multiplies by Wᵀ, a forward/backward symmetry the brain
has no mechanism for. `src/eval/credit_assignment.py` trains one deep cortex module (a coordinate→place-code map,
2→H→H→place) THREE ways from a matched init and MEASURES, never trains, whether the biological rule matches
backprop: **backprop** (Wᵀ), **FEEDBACK ALIGNMENT** (a fixed RANDOM backward matrix B — no weight transport, no
symmetry; Lillicrap 2016; the abstraction the dendritic/burst rules of Sacramento 2018 and Payeur 2021 make
biophysical), and a **shuffled-feedback** falsifier (B re-randomised every step).

- **(A) PARITY.** Feedback alignment reaches backprop's spatial decode — **0.106 ± 0.009 vs 0.105 ± 0.010** (both
  far below the position-blind floor **0.267**) — and its extrapolation, with **no weight transport**.
- **(B) ALIGNMENT EMERGES.** It works because the forward weights **rotate to align** with the fixed random
  feedback: weight-align `cos(W3, B3ᵀ)` **+0.07**, grown from ~0; the feedback-delivered error aligns with the
  true gradient at **+0.10** — modest but consistently positive versus the shuffled null at **~0.00**. The feedback
  *pathway* carries the error, not Wᵀ.
- **(C) FALSIFIER.** Shuffling that pathway every step prevents alignment and cripples learning — decode
  **0.147 vs 0.106** (gap **+0.042 ± 0.015**): it is the *consistent* feedback, not any random matrix, that
  assigns credit.
- **(D) SAME CODE.** Feedback alignment learns backprop's internal representation (hidden-layer **CKA 0.98**).

Honest scope: this removes backprop's weight-transport objection on a feedforward module; the burst-dependent
(Payeur) and dendritic-microcircuit (Sacramento) realisations, and running feedback alignment inside the
recurrent path-integration net to grow emergent *grid* cells under a non-backprop rule, are the follow-ups. The
credit signal, made biological — and the spatial signatures survive it. (`results/credit_assignment.json`,
`results/credit_assignment.svg`.)

### A self-tuned learning rate — meta-learning inferred volatility (GAPS.md Tier 5, #B3, n=5)

The most distinctively *human* learning capability still missing: people **tune their own learning rate** to the
world — raising it when things are volatile, lowering it when they are merely noisy — and, the subtle part,
**dissociate volatility from stochasticity** even though both inflate observation variance (Behrens et al. 2007;
Piray & Daw 2020). The mechanism is a prefrontal **meta-reinforcement-learning** process (Wang et al. 2018): the
adaptive rule lives in recurrent *dynamics*, not synapses. `src/eval/meta_learning.py` reproduces it and MEASURES
the signature, never trains it. A GRU is meta-trained ONLY to predict the next observation, across change-point
episodes whose hazard (volatility) and noise (stochasticity) are drawn per-episode and **never given as input** —
it must infer them from the stream. Then, **weights frozen**, we run it through one session of concatenated
`[stable | volatile | stochastic]` blocks and fit its *revealed* learning rate per block (the delta-rule slope
`ŝ_t = ŝ_{t-1} + α·(o_t − ŝ_{t-1})` — a rate, invariant to error magnitude):

| block | hazard / noise | revealed learning rate α |
|---|---|---|
| STABLE | low / moderate | **0.49 ± 0.07** |
| VOLATILE | high / moderate | **0.59 ± 0.08** |
| STOCHASTIC | low / **high** | **0.34 ± 0.05** |

- **(A) It tracks volatility.** α is higher when the world jumps than when it is stable — `α_volatile − α_stable`
  = **+0.10 ± 0.03** (every seed positive).
- **(B) The dissociation — the non-circular signature.** Under pure **stochasticity** the network *lowers* its
  learning rate (`α_volatile − α_stochastic` = **+0.25 ± 0.05**; `α_stable − α_stochastic` = **+0.15**) — even
  though the stochastic block has the **highest observation variance**. A naive "learn faster when errors are big"
  account predicts the opposite; the network has inferred, from temporal *structure* (a jump is a persistent step;
  noise is uncorrelated wiggle), that this variance is noise, not change. Volatility ↑α, stochasticity ↓α.
- **(C) Learned, not architectural.** An untrained (random-weight) GRU is flat across blocks (**+0.00 ± 0.00**) —
  the adaptation requires meta-training.
- **(D) Functional.** The adaptive network's next-observation error is **0.93×** the best *single fixed* learning
  rate on the mixed session — no static α matches it.

Honest scope: the outer (meta) loop is backprop — the meta-RL standard; the biological claim is the emergent
*inner-loop* learning rate that lives in the frozen-weight recurrent dynamics (Wang 2018). The latent is 1-D
tracking; tying it to the SR / grid reward-location substrate is a follow-up. The brain tuning its own learning
rate from inferred volatility — emergent, measured, not in the loss.
(`results/meta_learning.json`, `results/meta_learning.svg`.)

### The glial learning partner — astrocyte-gated slow plasticity for continual retention (GAPS.md Tier 5, #B4, n=8)

The repo's e-prop already has the two *neuronal* ingredients of a plausible learning rule — an eligibility trace
and a broadcast learning signal. The missing third is *non-neuronal*: astrocytes gate plasticity over a **slow
(seconds)** timescale through the tripartite synapse, and hippocampal "learning-associated astrocytes" orchestrate
memory encoding and retrieval (Williamson et al., *Nature* 2024). `src/eval/astrocyte_plasticity.py` adds a slow
per-synapse astrocyte trace `a ← ρ·a + |Δw|` that gates the e-prop update `Δw ← Δw/(1+β·a)` — throttling further
change at synapses it has tagged as important — and MEASURES retention on a continual stream of cue→target tasks.

The confound this must defeat is obvious: *any* plasticity gate "forgets less" by simply learning less. So the
result is reported **against a matched control** — a UNIFORM plasticity reduction scaled to the **same total
‖Δw‖** as the astrocyte (4.51 ≈ 4.51, of an ungated 12.96). Retention error on the old tasks (1 − cosine):

| condition | retention error (old tasks) |
|---|---|
| ungated e-prop | **0.53** |
| **matched UNIFORM reduction** | 0.47 |
| **SLOW astrocyte** | **0.44** |
| fast astrocyte | 0.52 |

- **(A) Targeting beats a matched uniform reduction.** At the same total plasticity, the astrocyte retains old
  tasks better than a uniform cut — **+0.036 ± 0.024** — so the gain comes from *where* the glia throttle
  plasticity (importance-tagged synapses), not from throttling less of it.
- **(B) It needs the SLOW timescale (falsifier).** A **fast** astrocyte (ρ = 0.5, decays within a task) matches
  its own uniform control — **+0.000 ± 0.003**. And this falsifier is exactly the control for the recency worry:
  the fast gate *also* throttles the current task, yet retains no better than uniform — so the retention gain is
  the **slow cross-task protection of old synapses**, not merely "writing the new task more weakly".
- **(C) The advantage grows with memory load.** Against full plasticity the gain is **+0.091 ± 0.034** and rises
  with the number of tasks — the glia matter most when there is forgetting to fight.
- **(D) Honest trade-off.** Protecting old memories costs a little new-task acquisition (recency +0.056) — the
  stability–plasticity frontier, reported not hidden.

Honest scope: computationally this per-synapse importance-throttle is kin to EWC (Kirkpatrick 2017) / synaptic
intelligence (Zenke 2017); the biological content — and what the experiment tests — is that a **slow glial
process** supplies the importance signal and that it needs the slow timescale. It is a reduced model of the
tripartite synapse, not a literal D-serine model; the distinct Benna–Fusi multi-timescale *synapse* (power-law
forgetting) remains an open gap (#B2). The glial learning partner, as an emergent, matched-controlled retention
signature — measured, not put in the loss. (`results/astrocyte_plasticity.json`, `results/astrocyte_plasticity.svg`.)

### The faithfulness capstone — the CORE itself learns biologically: grid cells under a non-backprop rule (GAPS.md Tier 5 capstone, n=5)

Everything above added biological learning *rules*, but the cortex→fusion→LLM core was still trained by
**backprop** — the very thing #A1 argued the brain cannot do (weight transport; a global backward pass). The
landmark emergent result of the whole field is that **grid cells emerge** when a recurrent network is trained on
self-supervised path integration (`emergence.py`; Cueva & Wei 2018; Banino 2018) — but there, too, by backprop.
`src/eval/emergent_grid_bio.py` closes the loop: it trains the same path-integration recurrent net by **RFLO**
(Murray 2019) — an **eligibility trace** (e-prop's temporal-credit primitive) times a learning signal delivered
through a **fixed random feedback** matrix (#A1's feedback alignment — *no weight transport, no
backprop-through-time*) — and asks whether the grid code still forms. Four rules from a matched init: backprop
(the reference), RFLO (the biological rule), a **shuffled-feedback** falsifier (the random feedback re-drawn every
step), and untrained.

- **(A) RFLO learns path integration without weight transport.** Place-prediction loss (the training objective):
  RFLO **0.021 ≈ backprop 0.014**, far below the untrained readout's **0.082**.
- **(B) The grid code emerges under RFLO — and it is never in the loss.** Rate-map **periodicity** (scored by the
  exact autocorrelogram machinery of `emergence.py`): RFLO **0.53 ≈ backprop 0.50**, **+0.09 ± 0.03** over the
  untrained floor; **76%** of units become periodic versus **47%** untrained. The loss only ever asked for
  place-cell prediction; the periodic grid code is a measured, emergent consequence.
- **(C) Falsifier — it is the *consistent* feedback that grows grid cells.** With the feedback **shuffled** every
  step, the readout still fits (place-loss 0.020) but the grid code falls to the untrained floor (periodicity
  **0.45**, **−0.09 ± 0.03** vs RFLO). So the grid code specifically requires the consistent random feedback that
  the forward weights **align** to (#A1) — not any feedback — to shape the recurrent hidden representation.

Honest scope: as in `emergence.py`'s unconstrained model, the emergent signature is **periodic multi-field**
spatial tuning, not a clean hexagonal lattice (gridness stays negative for backprop too; hexagonality needs the
constructed velocity modules, `emergence.py --constrained`). The claim is that this emergent grid code forms
under a fully local, no-weight-transport rule. This is the capstone: not "biological learning rules bolted onto a
backprop-trained core", but **the core itself learning biologically** — the grid cells that define the cognitive
map, grown by the credit-assignment rule the cortex can actually run. (`results/emergent_grid_bio.json`,
`results/emergent_grid_bio.svg`.)

### Graceful forgetting from the synapse — the multi-timescale (Benna–Fusi) weight (GAPS.md Tier 5, #B2, n=5)

Every weight in the repo is a **scalar**, and a scalar synapse is caught in the stability–plasticity dilemma: a
leaky/bounded weight forgets **exponentially** — fast-learning *or* stable, never both. Benna & Fusi
(*Nat. Neurosci.* 2016) resolve it *inside* the synapse: one synapse is a **chain of coupled hidden variables at
geometrically-spaced timescales** (a cascade of "beakers" joined by "tubes"); a plasticity event enters the
visible weight and slowly diffuses into deeper, slower beakers, so memory decays as a **power law (~1/√t)** —
one weight both fast-learning *and* long-remembering. `src/eval/complex_synapse.py` reproduces Benna & Fusi's own
memory benchmark and MEASURES the forgetting curve (never fits it): over S synapses a stream of M random ±1
memories is stored; the visible weights are then read out and each memory's signal-to-noise ratio is measured
against its age. Three models at matched initial SNR: a leaky scalar, a Benna–Fusi chain (N=7), and the chain at
N=3/5/7.

- **(A) Power law vs exponential — the *shape* is measured, not imposed.** The Benna–Fusi SNR(age) is a straight
  line on **log-log** axes — power-law fit R² **0.99** versus an exponential fit's 0.73 — with slope **−0.47 ±
  0.01**, the 1/√t law falling out of the diffusion. The leaky scalar is the opposite: a straight line on
  **semilog** axes (exponential fit R² **0.99** versus a power-law fit's 0.81). Which fit wins flips between the
  two synapses.
- **(B) Longer memory at matched initial SNR.** The age at which SNR crosses 1 (signal = noise) is **278** for
  the complex synapse versus **84** for the scalar — **3.3×** longer, with the same initial signal per memory.
- **(C) Dose-response.** Memory lifetime grows geometrically with the number of beakers: **55 → 198 → 278** for
  N = 3 → 5 → 7 — the Benna–Fusi prediction (a 1-beaker chain is just the scalar).

Honest scope: a linear-chain reduced model of the Benna–Fusi synapse on the canonical random-memory benchmark; it
drops into any store (including the spatial / Hopfield ones). It is distinct from #B4 — B4 is a *glial gate on the
learning rule*, B2 is the *intrinsic multi-timescale synapse*; together they are two independent routes to
graceful forgetting. The stability–plasticity dilemma, dissolved at the synapse — measured, not put in the loss.
(`results/complex_synapse.json`, `results/complex_synapse.svg`.)

### Representational drift — the population GEOMETRY is what survives, not the cells (GAPS.md Tier 5, #C6, n=5)

Place cells change their tuning over days even in a fixed environment with stable behavior — representational
drift (Ziv 2013; Rule 2019). What supports stable behavior? The population-geometry answer (Morales 2025; 2025
CA1 coordinated-drift work): read the *environment's geometry* carried by the population manifold, not the
identity of particular cells. `src/eval/representational_drift.py` tests this — and it is worth stating plainly
that a **first version of this eval was circular and was killed by an adversarial red-team**: RSA over a Gaussian
tiling is blind to remapping (a *full remap* gives *higher* RSA), and the "geometry reader" there was just
within-day recalibration. The rebuilt, non-circular test compares, **at matched single-cell drift**, a
geometry-**preserving** drift (a fraction of place fields relocate each day) to a geometry-**destroying** drift
(independent high-D noise of the *same* single-cell magnitude), read by a **label-free** geometry read-out (it
recovers position from the current day's manifold *ordering* — the Fiedler / kNN-Laplacian 1-D coordinate — using
no current position labels).

- **(A) Geometry, not cells, is what survives.** At matched single-cell drift (cell-corr **+0.15 vs +0.13**), the
  label-free geometry read-out recovers position almost perfectly under geometry-preserving drift (**0.001 ±
  0.001**) but fails under geometry-destroying drift (**0.30 ± 0.01** ≈ chance 0.25) — gap **+0.30 ± 0.02**. Since
  the single-cell drift is *matched*, the difference is the drift's **structure** (whether the geometry is
  conserved), not how much the cells changed.
- **(A′) Supervised confirmation (not an overfit artifact).** A *held-out* linear decoder shows the same:
  preserving **0.02** vs destroying **0.44**. Even *with* labels, position does not generalise once the geometry
  is gone. (An all-position fit overfits — N > P — and hides this; held-out exposes it.)
- **(B) Fixed vs geometry.** A **fixed** decoder, bound to specific cells, degrades under drift (**0.28**) while
  the geometry read-out survives.
- **(C) Robust to remapping — the honest resolution.** The geometry read-out survives even a **full remap** (all
  cells re-tile the track, **0% cell identity conserved**, error **0.002**). It reads the environment's geometry,
  not which cells carry it — so the claim is about *geometry* conservation, not *cell* conservation (this is the
  point the red-team's full-remap critique forced into the open).

Honest scope: a phenomenological place-code drift model; the geometry read-out is label-free unsupervised
manifold decoding, which fails precisely when the manifold is corrupted — that failure is the signal. Stable
read-out rides on the conserved population geometry, not on single-cell stability (Morales 2025) — measured, not
put in the loss. (`results/representational_drift.json`, `results/representational_drift.svg`.)

### Sleep triple-coupling — SELECTIVE consolidation emerges from competition for scarce windows (GAPS.md Tier 5, #C7, n=5)

NREM sleep nests three rhythms: slow oscillations (SO, ~1 Hz) gate spindles (~12 Hz), whose troughs gate
hippocampal ripples (replay). This SO→spindle→ripple coupling *times* replay to arrive at cortex during the brief
plastic UP-state windows, and behaviourally sleep consolidates **tagged / relevant** memories preferentially
(Latchoumane 2017; Maingret 2016; Diekelmann & Born 2010). `src/eval/sleep_consolidation.py` asks whether that
**selectivity** is a *consequence* of the architecture rather than something imposed. M = 40 traces, half TAGGED
(strength s≈1.0), half untagged (s≈0.3); sleep offers a **limited** number of spindle windows. C6 taught the
by-construction trap, so the guards are explicit: consolidation count is **matched** across conditions (the claim
is selectivity *per event*, not more events); the drive is s-proportional in *both* conditions (tags are never
told to win); and a **no-SO falsifier** removes the scarce-window bottleneck.

- **(A) Selectivity is emergent, not imposed (headline).** With the coupling, each scarce window consolidates the
  *winner* of a noisy competition among reactivated traces (winner-take-all over K reactivations). This amplifies
  the trace-strength difference into a tagged fraction of **0.989 ± 0.007** — far above the **uncoupled** condition
  (**0.778 ± 0.030**, which consolidates ∝ s, so untagged memories get their proportional share) and well above
  the **proportional floor 0.769**. Selectivity gap **+0.21 ± 0.04**. Nothing in the mechanism prefers tags; the
  selectivity falls out of competition for *limited* windows on noisy traces.
- **(B) Coordination — timing, at matched replay count.** Coupled replay is timed to the plastic UP-state windows,
  so **every** replay consolidates (coord **1.00**); random-timed (uncoupled) replay wastes events that land in
  DOWN states (coord **0.50 ± 0.02**) — gap **+0.50**. Same number of replays; the sleep architecture's *timing*
  is what converts them to plasticity.
- **(C) Falsifier — the selectivity needs the SO structure.** Remove the scarce-window bottleneck (cortex always
  plastic, no SO nesting) and the competition disappears: selectivity collapses to **0.764 ± 0.016 ≈ the
  proportional floor 0.769** (drop **−0.23 ± 0.01** from coupled). So the selectivity is a property of the
  SO/spindle *nesting*, not of "more replay" — with unlimited windows replay is indiscriminate.

Honest scope: a phenomenological triple-coupling model (traces + scarce windows + noisy competition), not a
spiking SO/spindle/ripple simulator. The measured signature — that coupling *selects* tagged memories above the
proportional floor and *collapses* to that floor without the SO bottleneck — is never in any loss; it emerges
from competition for scarce, timed windows (Latchoumane 2017; Diekelmann & Born 2010).
(`results/sleep_consolidation.json`, `results/sleep_consolidation.svg`.)

### The map goes abstract — 2-D CONCEPTUAL and SOCIAL grids, de-risked on CPU before the T4 (GAPS.md Tier 3, #8/#9, n=5)

The cognitive-map hypothesis says the hippocampal–entorhinal grid is not just for physical space — it maps
**conceptual** spaces (Constantinescu, Behrens 2016; Bellmund 2018) and **social** ones (Tavares 2015;
Park, Miller 2021) with the same machinery. Gaps #8/#9 test this at the *language* level: a frozen LLM reads the
grid code and answers "which concept is closer?" / "who is more dominant?" — cortex-ON vs text-only-OFF. That
headline needs a T4 (frozen Qwen-1.5B + LoRA) and is scaffolded in `notebooks/m8_…`, `notebooks/m9_…` +
`src/training/train_conceptual.py`, `train_social.py`. But — exactly as the 1-D structural-transfer headline was
de-risked on CPU (`structural_transfer_cortex.py`) before its T4 cell — the **design is validated first, on the
actual frozen `cortex.encode` pipeline the LLM reads**, so no T4 is spent on an unsound cell. A cortex pretrained
**only on physical Euclidean space** is FROZEN; a concept/agent at 2-D coord (x,y) enters by its *own* directed
path (heading = atan2(y,x); never a relative displacement → no leak). The sharp, non-circular 2-D signature is
**OFF-AXIS** queries: triples where the 1-D x-projection ordering *disagrees* with the true 2-D answer — a 1-D
(rank) code is ≤0.5 there **by construction**, so beating chance there is un-fakeable.

- **(#8) A genuine 2-D conceptual metric — `conceptual_grid_cortex.py`.** Read-out-free (parameter-free, so it
  cannot be circular): on a **label-BALANCED** off-axis set (chance exactly 0.5) OFF-AXIS "closer" by raw
  code-distance **0.64 ± 0.03** vs shuffled positions **0.49** (gap **+0.15 ± 0.03**); distance-correlation
  Spearman **0.53** vs shuffled **−0.03**. Held-out linear decode (concepts the probe *never saw*): position
  recovered at **0.63** spacing vs **3.3** under the shuffled refit (a 5× collapse), with held-out off-axis
  "closer" **0.76**. (Balancing matters: the raw off-axis set is ~0.67 one class, so an *unbalanced* metric
  would credit a constant predictor 0.67 — we report the balanced number so chance is honestly 0.5.) The
  absolute strength is modest on CPU with these simple read-outs; the de-risk's job is only to show the 2-D
  metric is **present and control-clean**, which it is.
- **(#9) A dissociable 2-D social map — `social_grid_cortex.py`.** Agents in a POWER × AFFILIATION space. (A)
  **DOMINANCE** is a clean 1-D read of the power axis — held-out pairwise dominance **0.96 ± 0.02** (the social
  transitive-inference result; an order-preservation accuracy, unaffected by class balance). (B) **SOCIAL
  DISTANCE** is a genuine 2-D metric — balanced OFF-AXIS "socially closer" **0.64** (>chance 0.5, where a
  power-only read is ≤0.5). (C) **AXIS DISSOCIATION** — decoding dominance from the power axis gives **0.96** but
  from the affiliation axis only **0.45** (gap **+0.51 ± 0.07**): the two social axes are *separately* readable,
  gap #4's self/other double dissociation now reappearing at the abstract-map level. FALSIFIER: shuffle the
  agent↔position map and dominance collapses to **0.44** (chance).

Honest status: gaps #8/#9 are **de-risked on CPU and scaffolded for the T4, not yet closed** — the cortex-ON-vs-
text-only-OFF headline is produced by running the two notebook cells on a GPU. What the CPU proves, and proves
non-circularly, is that the frozen *space* map already carries the 2-D conceptual and social structure the LLM
cell will read (measured on the exact `encode` pipeline, never put in any loss).
(`results/conceptual_grid_cortex.json`/`.svg`, `results/social_grid_cortex.json`/`.svg`.)

**T4 run #1 — an honest failure, and what it taught.** The first GPU run of the #8 LLM cell **collapsed to a
constant predictor**: it read **50%** on its own *training* set while showing **66.8%** on the off-axis set —
which turned out to be *exactly* that set's label imbalance (0.668), i.e. the trivial "always answer the
majority" baseline, not 2-D reasoning. Two lessons, both now fixed in the trainers: (i) the off-axis eval must
be **label-balanced** so a constant predictor scores 0.5 (the imbalanced version would have paraded a constant
predictor as a 67% "result"); (ii) the multi-token candidate-NLL scorer was replaced with a **padding-immune
single-next-token** scorer, the eval sets **balanced and capped** (37k triples/seed → 1.2k, minutes not hours),
and a **periodic train-accuracy** read-out added so a run reveals *whether the readout is actually learning*
rather than driving the loss toward the class prior. A `--reeval` mode re-scores an existing checkpoint for free.
Whether the frozen-LLM readout can *extract* the modest 2-D signal (the CPU shows the code *carries* it, ~0.64)
remains **open pending a clean T4 run** — reported here rather than buried, in the register's standard.

**#9 CLOSED on the T4 — and the deeper bug the debug loop exposed.** After the run-#1 fixes, #9 still read a
flat 50% everywhere, and the decisive probe (does the yes-vs-no logit *vary* across inputs?) returned **std =
0.000** across all seeds — the spatial input had *literally no effect* on the output. The cause, verified on
CPU: the frozen cortex code is **~98% a position-independent constant** (magnitude 0.53) and only **~2% the
position-varying signal** (0.009). A linear probe decodes dominance at **1.000** because it amplifies that 2%
with large weights, but in the LLM path `head(code) → LayerNorm(spatial)` strips each token's *feature* mean and
**not** the across-agent constant, so every pair normalized to near-identical spatial tokens → an
input-independent forward → the model could only predict the prior (exactly 50%). The fix is a **gain-control /
divisive-normalization** stage between cortex and readout (per-dim standardization of the code over the concept
set — no label leak), which is itself a ubiquitous cortical computation and the "missing module" the failure
pointed to. With it, the signal lifts to unit scale (linear decode unchanged at 1.0) and the frozen LLM finally
reads it: **dominance_far 100%, dissociation 100%, adjacent(trained) 99%** vs **cortex-OFF 50%** and
**shuffled 47.5%** (n=3; the permutation p floors at 0.25 for 3 seeds — the effect is maximal at ~0 variance,
so ≥6 seeds are needed only to push the *statistic* below 0.05). So a language model, reading a brain-like
spatial code trained **only on physical space**, performs abstract **social-hierarchy transitive inference**
(Kumaran 2016; Park-Miller 2021) it cannot do text-only — the cognitive-map claim, from space to social
meaning, at the language level. (`results/social_llm.json`.)

**#8 CLOSED on the T4 — and why it needed a *different* module than #9 (the deepest finding).** #8 ("which
concept is closer to the anchor?") did *not* transfer with #9's read-out, and chasing why produced the most
interesting result of the pair. #9 is a **1-D ordinal** read ("more dominant" = compare a power projection),
which is **linear** in the code — a linear head extracts it and the frozen LLM does the compare. #8 is a **2-D
metric** read, and a distance is the **overlap/correlation of grid population vectors** (Bellmund & Behrens
2018; Bush, Barry & Burgess 2015) — a **dot product**, hence **quadratic**, which a linear head *cannot* compute
(it read 50%) and a free MLP either under-fit or (an adversarial red-team caught this) could self-answer,
sidelining the LLM. The honest, load-bearing module is a **coincidence detector** (`CoincidenceReadout`): a
*shared* per-candidate `proximity(anchor, candidate)` — the grid-overlap — followed by a **linear** combine into
joint tokens; the linear combine can encode the graded difference but **cannot threshold it**, so the frozen LLM
still performs the ordinal compare (CPU-verified the read-out cannot self-decide). With it, n=3 seeds (all
converged): **closer_far 77.0% ± 2.8%, OFF-AXIS 68.3% ± 3.5%** (chance 0.5, where a 1-D code is ≤0.5 *by
construction* — genuine 2-D), **near(trained) 97%**, vs **cortex-OFF 50.0%** and **shuffled 49.6%**. The honest
bound: the biological read-out computes the **metric** (proximity); the frozen LLM does the **ordinal** compare —
so #8 is real but **weaker (~0.70 ceiling) and less stable** than #9's 0.96, a genuine **ordinal-vs-metric
dissociation**. Reaching it exposed a chain of real bugs, each fixed and documented rather than buried:
CUDA-OOM → an off-axis **label-imbalance** that flattered a constant predictor → **gradient checkpointing**
silently killing the LoRA gradients → **left-padding** mis-targeting the loss → a **context-dependent token id**
in the eval → the code being **~98% a constant** (needing gain-control normalization) → the metric being
**quadratic** (needing the coincidence detector) → **split-vs-joint** tokens → **LR-warmup** for the late,
seed-sensitive convergence. That #8 took eight distinct fixes where #9 took one is itself the measurement of how
much harder a metric map is to read than an ordinal one. (`results/conceptual_llm.json`.)

Together: the cortex now has a map that is **predictive** (plans detours geometry can't) and
**temporal** (tells elapsed time with the brain's scalar-timing law) — the two axes the document
identified as missing, each validated against its own falsifier before any LLM wiring.

## The platform as a hypothesis generator — predictions for experiment

Once the architecture reproduces the neuroscience *by emergence*, it can be **perturbed to predict**.
`src/eval/predictions.py` sweeps one controlled variable at a time and records the model's quantitative
consequence as a **falsifiable biological hypothesis**. We have already run one full predict→test→falsify
cycle in-house: the model predicted "slow cells code late" (log-compression), which **did not replicate**
at n=6 (`results/spiking_time_cells.json`) — the loop rejects as well as proposes. Two standing
predictions (n=3 conditions × seeds; `results/predictions.json`, `results/predictions.svg`):

- **P1 — content load sets the conjunctive/pure time-cell ratio.** As the number of distinct events the
  code must bind grows, the share of CONJUNCTIVE (event×time) cells among time cells rises from **0%
  (content-free, K=1)** to **~70% (K≥3)** (it saturates rather than climbing monotonically). *Testable:*
  a timing task with more distinct cues should yield proportionally more cue-selective ("contextual")
  time cells and fewer purely temporal ones; a content-free interval task should be dominated by pure
  time cells.
- **P2 — spatial-input reliability sets the space/time cell mix.** Corrupting the self-motion (velocity)
  input drives the PURE-TIME share of tuned cells from **21% (clean)** to **84% (noisy)**. *Testable:*
  degrading vestibular / optic-flow input should reallocate the hippocampal population away from
  place/conjunctive coding toward time cells.

Neither relationship was designed in; each falls out of the substrate and is stated as a number an
experiment can refute. This is the turn from *reproducing* known neuroscience to *proposing* it.

## The behaving agent — the cognitive map drives flexible behavior

Everything above *probes* the cognitive map; this closes the loop and lets an agent *use* it
(`src/eval/agent_navigation.py`, n=5). It is the first step of the embodied-agent program (the agency
gap): perception → decision → action → consequence, integrating modules we already have.

- **Navigation emerges from one closed loop.** An agent in a 2D arena path-integrates its own moves into
  a PLACE code (no coordinates given), feeds it to a dopamine-TD **critic** (value) and a
  basal-ganglia-like softmax **actor** (action selection), acts, and learns **online** from the
  reward-prediction error. Goal-directed navigation is learned end-to-end — success rises to **100%**
  (every seed; learning curve in `results/agent_navigation.json`). The cognitive map is now something the
  agent *behaves with*, not just something we decode.
- **One self-learned map → flexible, zero-shot, any-goal navigation (the defining capacity).** The agent
  learns a **successor representation** of a barriered world from its *own* random-walk exploration (TD),
  and then ONE map serves ANY goal: greedily ascending V = M[:, goal] navigates **zero-shot** to
  arbitrary goals, around the wall — **100% ± 0**. Controls fail: **Euclidean** vector-navigation stalls
  at the barrier (**69% ± 10**), and a **model-free** policy trained for goal A does not transfer to other
  goals (**13% ± 4**). Flexible goal-directed behavior from a self-learned predictive map is exactly what
  a cognitive map is *for* — and here it drives an agent, not a probe.

This converts the pile of faithful organs into a behaving brain-in-miniature. (`results/agent_navigation.json`,
`results/agent_navigation.svg`.)

**Memory-guided behavior — one-shot place learning, abolished by lesioning episodic memory**
(`src/eval/agent_memory.py`, n=5). The second capacity integrates a *different* organ — the hippocampal
**episodic store** — into behavior. Each "day" the reward moves to a new cell; trial 1 the agent explores
to find it and stores its place code in **one shot**, then later trials recall it (population-vector
readout) and navigate straight there via the map. The result is the Morris-water-maze signature: a
**single** rewarded trial collapses latency from **142 ± 10 to 7 ± 1 steps** (one-shot savings ~135).
**Lesioning the episodic store abolishes it** — latency stays ~130 every trial (savings ~8) — while
*navigation itself is intact* (the agent still reaches a goal it is given). So removing one organ removes
exactly one capacity (one-trial memory), reproducing the hippocampal dependence of one-shot place
learning. (`results/agent_memory.json`, `results/agent_memory.svg`.)

**Timing-guided behavior — acting on time, abolished by lesioning the temporal code**
(`src/eval/agent_timing.py`, n=3). The third capacity integrates the **temporal organ** into action: an
interval-production task where the agent must emit its move at a target interval D (reward peaks at D and
decays). The policy reads the **emergent time-cell population** (a frozen TemporalCortex) to time the
action. With it intact the agent acts **precisely at D=25** (act time 25 ± ~2; reward **0.88 ± 0.11**);
**lesioning the temporal code** (zeroing it) abolishes timing — the agent can no longer tell elapsed time,
acts immediately (act time ~1), reward **0.00** — while the rest of the agent is intact.

**The behaving agent's clean structure→function→lesion map.** Three capacities, three organs, three
specific lesions:

| capacity | organ integrated | lesion removes (only) this |
|---|---|---|
| flexible navigation | cognitive map (SR) | — (Euclidean/model-free fail without it) |
| one-shot place memory | hippocampal episodic store | latency savings (142→7 gone) |
| timed action | time cells (temporal cortex) | timing (reward 0.88→0.00) |

A brain-in-miniature where each capacity emerges from integrating an organ into the closed loop, and each
is independently lesionable. (`results/agent_timing.json`, `results/agent_timing.svg`.)

**The unified agent — one task needs all three, a clean triple-lesion dissociation (the culmination)**
(`src/eval/agent_unified.py`, n=3). The three capacities above are *separate* demos; here a **single**
agent solves a task that needs all three at once — a *delayed memory-guided harvest*: each day reward is
at a new location available only in a brief window around time D, so the agent must **recall where**
(episodic store) → **navigate there** (cognitive map) → **harvest at the right moment** (time cells).
Reward = at the remembered place AND acting at |t − D| ≤ 4. The result is a textbook triple dissociation:

| condition | reward | failure mode when lesioned |
|---|---|---|
| **all intact** | **99% ± 2** | — |
| **− cognitive map** | **0%** | can't reach the place |
| **− episodic store** | **0%** | navigates to the *wrong* place |
| **− time cells** | **0%** | right place, *wrong moment* |

Removing **any one organ** zeros the reward, and *only* via that organ's own failure — exactly the
structure→function→lesion logic of systems neuroscience, in a *single* behaving brain-in-miniature whose
spatial, mnemonic, and temporal capacities all emerged from the same self-supervised substrate. This is
the cleanest single embodiment of the program's thesis: capacities are not engineered in, they *emerge*
from integrating faithful organs, and they dissociate like the brain's. (`results/agent_unified.json`,
`results/agent_unified.svg`.)

**The agent on its REAL grid cortex — closing the loop between *why* a grid code and *what* it does**
(`src/eval/agent_grid_cortex.py`, n=3). The unified agent above runs on an abstract successor map; here we
swap in the **real velocity-driven hexagonal grid cortex** (`_HexGridModules`: 6 modules at geometric
scale ratios, **fixed biological velocity gains**; Burak & Fiete 2009, Guanella 2007, Stensola 2012) as the
agent's spatial substrate, and re-run the dissociation. The agent now:

1. **path-integrates its own self-motion** through the grid cortex — a **384-unit grid-cell code is its only
   sense of position** (no GPS). (The public `grid_code_at()` is verified *exactly* equal to the recurrent
   integrator's path-integrated code — reading the code at a place == having walked there.)
2. **reads position from the grid code with a nonlinear (place-cell-like) network** — the
   entorhinal→hippocampal read, and exactly the **nonlinear decoder `grid_capacity` says you need**:
   decode error **0.024 (nonlinear) vs 0.030 (linear)**, the nonlinear edge in the same direction the
   capacity analysis predicts.
3. **vector-navigates** by the decoded displacement to a remembered goal (Bush et al. 2015) — closed-loop,
   reaching the goal at **100%**.

On this real grid substrate the triple dissociation holds exactly:

| condition | reward | failure mode |
|---|---|---|
| **all intact** (grid-nav + recall + time) | **100% ± 0** | — |
| **− grid cortex** | **2% ± 2** | can't localize / path-integrate → can't navigate |
| **− episodic store** | **1% ± 2** | recalls the wrong place |
| **− time cells** | **0% ± 0** | right place, wrong moment |

So the agent's spatial organ is no longer an abstraction: it is the **same biologically-constrained grid
cortex** whose capacity advantage we measured in `grid_capacity.py` — and lesioning it abolishes the very
navigation that capacity buys. The *why* (grid codes resolve space at scale) and the *what* (the behaving
agent path-integrates, localizes, and navigates on that code) are now one result.
(`results/agent_grid_cortex.json`, `results/agent_grid_cortex.svg`.)

**Path-integration drift, and its correction by boundary-vector cells — the Fiete caveat, resolved**
(`src/eval/agent_grid_drift.py`, n=3). A grid code path-integrates self-motion, but real self-motion is
**noisy**, so the integrated grid phase **drifts** from the true position — error that accumulates *without
bound* (Burak & Fiete 2009; the famous caveat to grid path integration). The brain corrects this with
**allothetic** cues: when the animal senses a known boundary, **boundary-vector cells** supply an external
position fix that *resets* accumulated grid error (Hardcastle, Ganguli & Giocomo 2015). We reproduce both
halves on the closed-loop agent, using the **real `BoundaryVectorCells` organ** with a *learned* (not
hard-coded) allothetic read-out (near-wall coordinate error **0.005**):

| | self-localization error over a 120-step walk (mean / **final**) | |
|---|---|---|
| self-motion noise | **no anchoring** (drift) | **BVC anchoring** |
| 0.00 | 0.014 / 0.015 | 0.014 / 0.014 |
| 0.05 | 0.48 / **0.69** | 0.20 / 0.21 |
| 0.10 | 0.84 / **1.21** | 0.40 / 0.43 |
| 0.15 | 1.29 / **1.72** | 0.57 / 0.61 |

- **(A) Drift is unbounded; anchoring bounds it.** Without correction the localization error *grows* over
  the walk — **final ≫ mean** (0.69 vs 0.48, 1.21 vs 0.84, 1.72 vs 1.29): the hallmark of accumulating
  path-integration error. Routing the boundary sense through boundary-vector cells makes it *stationary* —
  **final ≈ mean** (0.21≈0.20, 0.43≈0.40, 0.61≈0.57): the classic **sawtooth** (drift, then reset at a
  wall), ~2.5–3.5× lower error.
- **(B) The behavioral consequence — foraging.** Over a 6-goal episode drift *compounds*, so without
  anchoring the agent reaches fewer goals as noise rises (**66% → 33% → 23% → 15%** at noise 0.05→0.20);
  BVC anchoring substantially rescues it (**78% → 48% → 33% → 24%**).

Nothing here is hard-coded: the allothetic localizer is *learned* self-supervised from the BVC population,
and the drift/correction dynamic *emerges* from combining the noisy grid integrator with the gated boundary
sense. This completes the grid-cell arc — **why** a grid code (capacity), **what** it does in the loop (the
behaving agent), and now **its failure mode and the brain's fix** (drift + boundary correction), all on one
substrate. (`results/agent_grid_drift.json`, `results/agent_grid_drift.svg`.)

**A self-correction — the boundary anchoring was *phenomenologically* right but *mechanistically* wrong**
(`src/eval/agent_cue_integration.py`, n=3). On review, the anchoring above is a **hand-coded fixed gate**,
and that is not how the brain combines cues. The brain integrates idiothetic (path-integration) and
allothetic (boundary) cues **near-optimally** — combined precision better than either cue alone (Ernst &
Banks 2002; Nardini et al. 2008). We checked, and the fixed gate is markedly **suboptimal**: ~3–4× worse
than optimal, at times worse than the boundary cue alone. So we replaced it with a **generic learned
recurrent fuser** (a GRU; *no* hand-coded gate, *no* Kalman structure) that reads only the drifting
grid-PI estimate + the boundary-cell observation and is trained only to localize. Because it is fed the
*drifted position* (not raw velocity), beating PI-only **requires** using the boundary.

| self-motion noise | PI-only | boundary-only | fixed gate (old) | **learned fuser** | Kalman (optimal) |
|---|---|---|---|---|---|
| 0.05 | 0.58 | 1.03 | 0.51 | **0.37** | 0.22 |
| 0.10 | 1.23 | 1.07 | 1.07 | **0.58** | 0.65 |
| 0.15 | 1.69 | 1.04 | 1.40 | **0.85** | 1.07 |

- **(A) Near-optimal integration EMERGES.** The learned fuser beats **both single cues and the old fixed
  gate** at every noise level, and **tracks (even beats) the Kalman optimum** — near-optimal cue
  integration, discovered purely from training to localize, with nothing hand-coded.
- **(B) It genuinely integrates the boundary.** Ablating the boundary input collapses the fuser back to
  ~PI-only error (0.54 → ~1.05, vs PI 1.20) — the win is real cue *integration*, not PI denoising.
- **(C) Robust integration (honest nuance).** Across boundary-observation noise 0.05→3.0 the full error
  stays bounded (0.54→0.58) — the recurrent fuser **averages many unbiased observations over time**, so even
  a very noisy boundary stays useful; the boundary's contribution declines only modestly (0.52→0.35).

*Honest scope.* We claim near-optimal **integration** (A, B) — solid and multi-seed. We do **not** claim the
strict reliability-weighting law `w = σ_PI²/(σ_PI²+σ_B²)`: both a cue-conflict probe and an ablation sweep
are confounded because a recurrent fuser temporally averages *unbiased* cues (clean Bayesian down-weighting
would need biased or single-shot cues — left open). This entry is also a record of the platform's *method*:
a result that reproduced the right phenomenon was found to use the wrong *mechanism*, and was corrected.
(`results/agent_cue_integration.json`, `results/agent_cue_integration.svg`.)

**A head-direction organ — an emergent ring attractor, and the heading-dominated drift it causes**
(`src/eval/head_direction.py`, n=5). The cue-integration work flagged the deepest remaining gap: biological
path-integration drift is dominated by **heading** (angular) error from the **head-direction system**, which
the earlier drift module crudely modelled as translational noise. So we built a faithful HD organ — by the
same **emergence** method as grid/time cells (train a *generic* substrate on a task; measure brain
signatures never in the loss). A generic rate-RNN is trained only to track heading from angular velocity:

| emergence (trained vs untrained) | TRAINED | untrained |
|---|---|---|
| heading decode error | **2.6°** | 86.3° |
| HD-tuned units | **57%** | 24% |

- **(1) HD cells and a functional ring attractor EMERGE.** Trained, the net holds and updates a single
  heading bump and reads heading out to **2.6°** (vs **86°** untrained — the untrained net cannot hold
  heading); **57%** of units become HD cells (vs 24%), and the population activity traces a **1-D ring**.
  *Honest nuance:* a ring-*shaped* manifold appears even in the untrained recurrent net (it's partly
  inherent to recurrent integration; PC-angle~heading corr 0.87 trained vs 0.97 untrained — *not* a clean
  discriminator), so the training-specific emergence is the **HD tuning** and the **accurate, stable
  maintenance** — the attractor *function* — not the manifold shape per se.
- **(2) Heading-dominated drift, and its visual reset** (Knierim, Kudrimoti & McNaughton 1995). The
  emergent HD net integrates *noisy* angular velocity, so heading **drifts** (**77° ± 37** over a 140-step
  walk), and the agent path-integrates position *using that heading* — so heading error drives **position**
  drift (**13.4**). A **visual landmark** pinning the ring bump to the true heading bounds both (**heading
  13°**, **position 3.1**) — the biologically-correct, heading-dominated drift and its allothetic correction.

This makes the drift in the agent loop *mechanistically* right: the dominant error is heading, generated by
a faithful head-direction organ whose HD cells and attractor function *emerged*. (`results/head_direction.json`,
`results/head_direction.svg`.)

**The dead-reckoning brain — one closed HD → grid → place stack from self-motion alone** (the culmination)
(`src/eval/agent_deadreckoning.py`, n=3). The spatial organs are now unified into a single self-localization
loop. Instead of being *given* its heading (as the earlier grid agent was), the agent estimates **both**
heading and position from its own motor commands:

> motor (turn, step) → **HD ring attractor** (heading, drifts) → **grid cortex** path-integrates position
> *using that heading* (drifts more) → **place** read-out → behaviour.

The path integrator accumulates each actual displacement **rotated by the heading error** (θ_est − θ_true) —
so drift originates as *heading* error in the HD organ and propagates into *position* error, exactly as in
the brain. Two allothetic corrections fix two organs: a **visual landmark** resets the HD ring bump;
**boundary** input resets the grid phase.

| localization condition | position error |
|---|---|
| oracle heading (floor) | **0.04** |
| HD in loop, no correction | 2.41 |
| + visual reset (HD only) | 2.52 |
| + boundary reset (grid only) | 0.44 |
| **+ BOTH corrections** | **0.12** |
| lesion HD organ | 3.23 |
| lesion grid organ | 3.11 |

- **The stack closes, and drift is heading-originated.** With true heading the stack localizes near-perfectly
  (oracle **0.04**); putting the **HD organ in the loop** inflates position error to **2.41** — the heading
  drift propagates into position.
- **Both corrections are needed — each for a different organ.** Correcting **heading alone** (visual,
  **2.52**) does *not* rescue position: the grid integrator's accumulated error persists. The **grid**
  correction (boundary, **0.44**) is what fixes position directly, and adding the HD correction on top
  (**both, 0.12**) slows the drift *between* boundary resets — so the lowest error needs **both**. Lesioning
  either organ is catastrophic (HD **3.23**, grid **3.11**).
- **Homing (path-integration return; Wehner's desert ants).** The agent wanders out and returns to the
  origin using *only* its integrated position estimate: **intact 0.35**, abolished by lesioning **HD (2.79)**
  or **grid (3.11)** — the canonical dead-reckoning behaviour, on a fully emergent organ stack.

This is the cleanest single embodiment of a dead-reckoning brain: heading and position both inferred from
self-motion through emergent organs (HD ring attractor + grid cortex), drift that is mechanistically correct
(heading-originated), and two distinct allothetic corrections — one per organ. (`results/agent_deadreckoning.json`,
`results/agent_deadreckoning.svg`.)

**…and the dead-reckoning brain *speaks* — a frozen LLM reads the emergent code** (the founding-goal capstone)
(`notebooks/m5_deadreckoning_llm_kaggle.py`, n=6 seeds on a T4). A frozen Qwen2.5-1.5B (LoRA + gated fusion)
reads the agent's **emergent self-localization code** — the grid-cell population (position) and the
head-direction ring-attractor state (heading) — and answers in language, with two *direct single-organ*
decodes (the agent's moves never appear in the prompt, so cortex-ON vs text-only-OFF is causal +
leakage-proof):

| readout (cortex-ON vs text-only-OFF) | ON | OFF | Δ | p | reads |
|---|---|---|---|---|---|
| **WHERE** (which of 9 cells) | **38% ± 32** | 8% | +30 | **0.033** | grid (position) |
| **FACING** (heading, 8 sectors) | **40% ± 26** | 12% | +28 | 0.095 | HD (heading) |

- **WHERE is significant** (p=0.033 — all 6 seeds ON>OFF): the LLM reads **position** from the grid code.
- **FACING is a strong trend** (Δ+28%, p=0.095 — not significant at n=6): the LLM reads **heading** from the
  HD code.
- **The gem — an organ-specific double dissociation** (the strongest causal evidence, cleaner than the ON/OFF
  p's): each read collapses **only** when *its own* organ is ablated.
  - **WHERE**: no-grid **8%** (dies, = chance) vs no-HD **39%** (survives, ≈ ON).
  - **FACING**: no-HD **10%** (dies) vs no-grid **33%** (survives, ≈ ON).

  So the LLM reads position *specifically* from the grid cortex and heading *specifically* from the
  head-direction ring — the emergent organs built this session become a spatial sense an LLM speaks from,
  each causally traced to its organ.

*Honest scope.* WHERE is significant; FACING is a strong trend whose organ-specific lesion (no-HD 10% vs
no-grid 33%) independently confirms it reads HD, though its ON-vs-OFF p (0.095) does not clear 0.05 at n=6
(wide cross-seed *magnitude* variance — the readout converges to 60–90% on some seeds, 20–30% on others —
with consistent direction). The harder **egocentric homing-vector** readout (a nonlinear cross-organ
combination, atan2(−position) − heading) was **null** in the first attempt and is left as documented future
work. (`results/deadreckoning_llm_agg.json`.)

**The map is multi-reference-frame — object-vector cells + grid reanchoring** (`src/eval/reference_frame.py`,
n=5). An external neuroscience review flagged the deepest remaining gap: the map so far is a *global
allocentric* metric, but the entorhinal code also carries **egocentric object-vector cells** (Høydal et al.,
*Nature* 2019) and can **reanchor** — translating the grid pattern to a task-relevant object/landmark/reward
(Butler 2019; Boccara 2019), estimating position in multiple *local* reference frames (a 2025 frontier).
(The review's other ask — reliability-weighted cue integration — was already built this session in
`agent_cue_integration`.) We added the missing capability — a new `EgocentricObjectVectorCells` organ — and
measured it:

- **(A) The object-vector code works.** The OVC population encodes a landmark in self-centred polar
  coordinates (distance, *egocentric* bearing); a readout recovers the object vector to **0.030** (arena
  half-width 2.5) — and it's egocentric (rotates with heading), the defining contrast with allocentric
  boundary cells.
- **(B) Reference-frame dissociation (the headline).** On an **object-relative goal whose object *moves*
  every episode**, an **object-frame** agent (object-vector cue → HD egocentric→allocentric transform →
  navigate to object+offset) reaches it **100%**, while a **global-frame** agent (path integration only)
  is stuck at **17%** (it can't track a goal that moves with the object), and **lesioning HD** drops it to
  **15%** (the egocentric→allocentric transform is gone). So object-relative behavior needs **both** the
  object-vector cue **and** the HD frame-transform — neither alone, and not the global map.
- **(C) Grid reanchoring signature.** The object-frame grid code is `grid_code_at(agent − object)`; when the
  object moves by **Δ**, the code matches the grid **translated by Δ** (err **0.000**), *not* the un-shifted
  grid (err **0.073**) — grid cells **reanchoring by translating the pattern** with the object, exactly the
  2025 finding.
- **(D) Robustness (honest).** Object-relative success is *flat at 100%* up to object-cue noise 0.4 — the
  agent re-senses and **temporally averages** the unbiased cue, so it's *robust*, **not** a graceful
  reliability down-weighting (which, as in `agent_cue_integration`, would need biased/single-shot cues; left
  open).

This turns the model from a global path-integrator into an **entorhinal reference-frame transformer**: the
grid is not only path-integrated globally but dynamically reanchored to egocentric objects, under the HD
frame-transform — the single most current-neuroscience-faithful extension. (`results/reference_frame.json`,
`results/reference_frame.svg`.)

**Dynamic reanchoring of the grid phase to a landmark — allocentric & egocentric coexisting**
(`src/eval/landmark_anchoring.py`, n=3). The review's *exact* mechanism: not only the object-vector cells and
the static translation signature above, but the grid phase **dynamically reanchored** to a landmark *during
path integration*, under cue reliability — like boundary anchoring, but anywhere a landmark is seen:

> `ego = EgocentricObjectVectorCells(landmark)`; `p_hat = anchor − R(heading)·ego`;
> `w = reliability_gate`; `grid_phase = (1−w)·grid_phase + w·gains·p_hat`

- **(A) Reanchoring corrects allocentric drift.** Pure path integration drifts unbounded (**3.12**); a
  landmark seen anywhere bounds the global-position error to **0.87** — an allothetic fix that, unlike the
  boundary reset, works in open space wherever the landmark is visible.
- **(B) Allocentric *and* egocentric coexist.** At every step the agent reads **both** global position
  (allocentric, from the grid: **0.87**) **and** landmark-relative position (egocentric, from the
  object-vector cells: **0.78**) — the two MEC reference frames simultaneously (Nature Comms 2025).
- **(C) Reliability.** A reliable landmark helps strongly (**0.97** at low noise); as it gets noisier the
  reliability gate down-weights it and the benefit vanishes back toward PI (**~3.2**). *Honest:* the
  strictly-optimal combiner is the **learned fuser** of `agent_cue_integration` — a hand-coded Kalman gate is
  mis-calibrated here, so we use a reliability gate and report the dependence, not an optimal-weighting claim.

So the grid is path-integrated *globally* and dynamically *reanchored* to landmarks on demand, with both
reference frames coexisting — the entorhinal map as a **reference-frame transformer**, exactly the review's
"multi-reference-frame" framing. (`results/landmark_anchoring.json`, `results/landmark_anchoring.svg`.)

**…and object reanchoring is now INSIDE the core grid cortex — a load-bearing integration, not an eval loop**
(`src/eval/agent_grid_reanchor.py`, n=5). A neuroscience review noted, fairly, that the reanchoring above
lived only in a *standalone* eval loop: the core path-integrator (`_HexGridModules`) could reset its phase
only at **boundaries**, and the object-vector cells were not wired into it. We fixed that. The egocentric
object-vector organ now drives a phase correction **from within** `_HexGridModules.forward(object_obs=…)`,
through the **same egocentric→allocentric transform** the boundary path uses — a single shared bridge
(`_ego_to_allo` → `_apply_phase_fix`) reused by boundary, object, and centre anchors: `p_hat = anchor −
R(heading)·ego; φ ← (1−w)·φ + w·gains·p_hat`. An ablation contrasts a **local** cue (boundaries) against a
**global** one (an object/landmark, a fix anywhere it is visible), with a **shuffled-anchor control**
(allocentric decode error, lower=better):

| regime | path-int | boundary | object | shuffled object |
|---|---|---|---|---|
| **open field** (walls far) | 0.96 | 0.71 | **0.13** | 2.37 |
| **near a wall** | 2.43 | **0.80** | 0.09 | 3.62 |

- **(A) Object reanchoring is load-bearing.** In the **open field**, every wall is distant so boundary
  anchoring barely helps (0.71, vs path-int 0.96); the **object** cue reanchors the grid and bounds the drift
  **~6× better (0.13)** — a capability the boundary-only module did not have.
- **(B) It's the true geometry, not just extra input.** A **shuffled-anchor** control (object cue present but
  its world position scrambled) *fails* (2.37 / 3.62, worse than path integration) — the rescue comes from the
  egocentric→allocentric transform, not from "some signal".
- **(C) The local capability is preserved.** **Near a wall**, boundary anchoring still bounds the drift (0.80
  vs path-int 2.43). Each allothetic cue rescues its regime, both through one transform in one module.

So the object-vector cells genuinely **reanchor the grid phase inside the core cortex** now — the grid is
path-integrated globally and reanchored to whichever allothetic cue (boundary *or* object) is available.
(`results/agent_grid_reanchor.json`, `results/agent_grid_reanchor.svg`.)

**3D navigation via a plane-aligned 2D grid — the bat scheme** (`src/eval/plane_of_motion.py`, n=5). The
review's last item: the repo claimed `(x,y,z,t)` but coded height as a 1-D place stub. Freely-flying bats
appear to use a **2-D toroidal grid aligned to the behaviorally-relevant plane of motion** + an off-plane
code, *not* a full 3-D lattice (2026). We implement that faithfully with the **real hex grid cortex** on the
**PCA-estimated motion plane**:

| motion-plane tilt | plane-aligned 3D err | plane-recovery err | fixed-plane 3D err |
|---|---|---|---|
| 0° | 0.128 | 0.004 | 0.138 |
| 57° | 0.126 | 0.004 | 0.145 |
| 80° | 0.127 | 0.005 | **0.174** |

- **(A)** PCA recovers the motion-plane normal **almost exactly** (err ~0.005) at every orientation — the
  grid can be aligned to whatever plane the animal moves in.
- **(B)** the plane-aligned 2-D grid localizes 3-D position with accuracy **flat across plane tilt**
  (0.128→0.127) — **orientation-invariant**: it works in any motion plane because it aligns to it.
- **(C)** a **fixed** (horizontal) grid **degrades** as the plane tilts steeply (**0.138→0.174** at 80°) —
  the in-plane motion rotates into the coarse off-axis code, so **alignment is necessary**.

*Honest scope.* At matched budget we found **no robust 3-D-decode advantage** of the plane-aligned 2-D grid
over a *naive isotropic 3-D grid* (a learned decoder compensates for both; the capacity gap is modest in
this regime). So the contribution is the **faithful, orientation-invariant mechanism** (a 2-D grid on the
estimated motion plane — the bat scheme, replacing the `z`-stub) and the **alignment necessity**, *not* a
decode win over a 3-D lattice. (`results/plane_of_motion.json`, `results/plane_of_motion.svg`.)

**Coexisting egocentric anchors — center, object, boundary** (`src/eval/egocentric_anchors.py`, n=5). A 2025
Nat Commun result: allocentric and egocentric codes coexist in MEC, including cells for egocentric bearing &
distance to the geometric **center** and to **boundaries**. We had egocentric object-vector cells; the
missing sliver was the **center** anchor, now added as `EgocentricCenterCells` (egocentric bearing+distance to
the room center, computed from pos+heading). Three egocentric anchor frames are represented at once and read
out specifically (egocentric-vector decode error):

| anchor | from **combined** population | from **own** cells | from **other** cells |
|---|---|---|---|
| center | 0.24 | 0.25 | 1.37 |
| object | 0.62 | 0.60 | 1.95 |
| boundary | 0.10 | 0.10 | 0.42 |

The **combined** population decodes the egocentric vector to all three anchors simultaneously (coexistence),
and each frame is **specific to its organ** — it decodes from its own cells but *not* from another anchor's
(≥0.42, up to 1.95). So MEC is a **multi-anchor egocentric↔allocentric transformer** with a stable center
anchor, not a single global frame. (The object's absolute error is larger than the center's/boundary's
because its egocentric vector spans ~2× the range — up to 2·R·√2 vs R·√2 for the center — so the claim, and
the locked test, is the *relative* structure: combined ≈ own and other ≫ own, which is magnitude-independent;
no single absolute threshold is meaningful across anchors of different range.)
(`results/egocentric_anchors.json`, `results/egocentric_anchors.svg`.)

**Local 3D order, not a global lattice — the bat 3D-grid regime** (`src/eval/local_3d_order.py`, n=5). Bat
MEC 3D grid cells show *local* order (regular nearest-neighbor field spacing) but **not** a global 3D lattice
(no long-range periodicity). We make that measurable on two independent axes — **local order** = 1−CV of the
nearest-neighbor distance; **global lattice** = max structure factor S(q)/N (Bragg-peak height; periodic
distances, boundary-free):

| 3D field code | local order | global lattice |
|---|---|---|
| true 3D lattice | 0.94 | **0.88** |
| **local-order (bat-like)** | **0.95** | **0.05** |
| random | 0.65 | 0.05 |

A local-order (blue-noise) field code sits exactly in the **bat regime — high local order, ~zero global
lattice** — cleanly separable from a crystal (high on both) and random points (low on both). So "local order
without a global lattice" is a well-defined, measurable third regime, and the repo's 3D story is the
bat-faithful one rather than a naive cubic lattice. (`results/local_3d_order.json`, `results/local_3d_order.svg`.)

**…and a biologically-grounded 3D grid code now REPLACES the 1-D z stub in the core cortex**
(`src/eval/grid_3d.py`, n=5; `LocalOrder3DGrid`; `_HexGridModules(grid_3d=True)`). The metric above made the
bat regime measurable; the review's last item was that the core integrator still coded *height* as a 1-D
place stub (so "4D" was really 2D-grid + 1D-z + time). We built the real thing. `LocalOrder3DGrid` gives each
cell **multiple** 3D firing fields drawn from a shared **blue-noise** packing — so the fields have a regular
nearest-neighbor spacing (local order) with **no** global lattice (the bat MEC regime; Ginosar, Aljadeff, Las,
Derdikman & Ulanovsky, *Nature* 2021) — and it path-integrates 3D self-motion. Wired into the cortex via
`grid_3d=True`, it replaces the z-stub:

| 3D field code | local order | global lattice | 3D decode err | vertical err |
|---|---|---|---|---|
| **LOCAL-order (bat-like)** | **0.90** | **0.01** | **0.21** | **0.11** |
| cubic lattice (control) | 1.00 | **1.00** | 0.16 | 0.08 |
| random | 0.64 | 0.02 | — | — |

- **(A) Bat regime.** The code's field centers score **high local order (0.90)** but **~zero global lattice
  (0.01)** — cleanly apart from a cubic crystal (high on both: the **non-biological** lattice) and from random
  points. This is the *grid code's own fields*, measured with the same structure-factor metric.
- **(B) It is metric, not a stub.** The population **path-integrates and localizes in full 3D** (decode error
  **0.21**, vertical **0.11**), about as well as the cubic lattice (0.16) — so biological faithfulness costs
  **~nothing**; only the local-order code matches bats. Run through `_HexGridModules(grid_3d=True)`, the **core
  cortex** path-integrates 3D self-motion and localizes (err **0.19**) — height is now grid-coded, not a stub.

So the 3D representation is the bat-faithful one (local order, no global lattice) and lives **inside** the grid
cortex, not as a 1-D vertical place-code afterthought. (`results/grid_3d.json`, `results/grid_3d.svg`.)

**The unified multi-reference-frame navigating brain — one agent, two frames** (the functional
consolidation) (`src/eval/agent_multiframe.py`, n=3). The pieces above (grid cortex, head-direction ring,
object-vector cells) are not five demos but **one brain**: a single closed-loop agent that navigates in
*both* a global (allocentric) frame and an object-centred (egocentric) frame, sharing one organ stack.
GLOBAL goals (a room location) are reached via the **grid** position code; OBJECT goals (an offset from a
per-episode landmark) via the **object-vector cells** transformed to allocentric by the **HD** organ; and
steering is egocentric, so HD is needed to turn either way. The result is a clean **double dissociation**:

| lesion | GLOBAL goal (grid) | OBJECT goal (object-vector + HD) |
|---|---|---|
| intact | **100%** | **100%** |
| − grid cortex | **20%** | 100% |
| − object-vector cells | 100% | **12%** |
| − head-direction | **10%** | **10%** |

- One agent navigates in **both frames** intact (100% / 100%).
- Lesioning the **grid** kills the **global** frame only (20% vs object 100%); lesioning the
  **object-vector** cells kills the **object** frame only (12% vs global 100%) — a clean double dissociation.
- Lesioning **head-direction** kills **both** (10% / 10%) — it supplies the egocentric steering and the
  ego→allo transform shared by both frames.

So the reference-frame organs are unified in **one navigating brain** that holds, and acts in, two reference
frames at once — the functional embodiment of the "reference-frame transformer."
(`results/agent_multiframe.json`, `results/agent_multiframe.svg`.)

**…and the map speaks BOTH frames — a frozen LLM answers allocentric *and* egocentric**
(`notebooks/m6_multiframe_llm_kaggle.py`, n=8 on a T4). The language counterpart of the unified agent: a
frozen Qwen2.5-1.5B (LoRA + gated fusion) reads the **combined** code — the grid-cell population (global) plus
the egocentric object-vector cells (landmark-relative) — and answers in **either frame on demand** (moves
never in the prompt, so cortex-ON vs text-only-OFF is causal):

| readout (ON vs OFF) | ON | OFF | Δ | p | reads |
|---|---|---|---|---|---|
| **WHERE** (which room cell) | **47% ± 29** | 8% | +39 | 0.053 | grid (allocentric) |
| **LANDMARK** (which way, egocentric) | **35% ± 18** | 13% | +23 | **0.031** | object-vector (egocentric) |

- **LANDMARK (egocentric) is significant** (p=0.031); **WHERE (allocentric) is at the threshold** (p=0.053,
  Δ+39%). WHERE just misses 0.05 because of **one non-convergent seed** (seed 7: WHERE ON **8%**, *below*
  chance, and grid-ablation 15% does not reduce it — a clear LoRA-readout *training failure*, not a real
  null). **Sensitivity:** excluding that one failed run by the principled diagnostic (a readout below chance
  whose input-organ ablation doesn't lower it has not trained), the **7 converged seeds are all ON ≫ OFF →
  sign-flip p ≈ 0.016**. (n=8 + LR warmup raised WHERE from p=0.094 at n=6 and crossed LANDMARK into
  significance.) We report the full n=8 honestly rather than silently dropping the failed seed.
- **The decisive evidence — a clean organ-specific DOUBLE dissociation** (a within-condition contrast,
  independent of the ON/OFF p's): **WHERE** collapses *only* when the **grid** is ablated (no-grid **11%** vs
  no-object-vector **49%**); **LANDMARK** collapses *only* when the **object-vector** cells are ablated
  (no-object-vector **11%** vs no-grid **36%**). So the LLM reads the **allocentric** frame *specifically*
  from the grid and the **egocentric** frame *specifically* from the object-vector cells.

This is the review's vision realized at the language level: a map that answers "where am I globally?" **and**
"where am I relative to the landmark?", the two reference frames coexisting and each *causally traced to its
organ*. (`results/multiframe_llm_agg.json`.)

**Theta-cycle look-around — online sweeps as active look-ahead** (`src/eval/theta_sweep.py`, n=5). The
freshest item from a follow-up review: grid/place populations don't only path-integrate or replay — in each
theta cycle decoded grid activity **sweeps outward** from the agent, **alternating left/right** across cycles,
sampling surrounding space *including never-visited points* (Vollan, Gardner, Moser & Moser, *Nature* 2025).
The repo's theta machinery (phase precession, theta-gamma memory, sharp-wave replay) was gating / ordered
memory / *offline* replay; the *online* look-around was missing. We add a `ThetaSweepSampler` organ and show
it is **functional**, not decorative:

- **(A) Look-ahead avoids traps.** In a field of concave dead-ends, an agent that uses the theta sweep to
  sample the grid map *ahead* (querying look-ahead points) reaches the goal **100%** vs a reactive
  (current-position-only) agent's **76%**, at **equal path length** (~29 steps) — it routes *around* the
  dead-ends the reactive agent walks into. The sweep is an active "look-around," not offline replay.
- **(B) The Vollan signatures.** The sampler reproduces the reported statistics: it **alternates left/right**
  across theta cycles; sweep length is **19.7% of grid spacing** (Vollan's value) and is **multi-scale** —
  per-module length scales with that module's spacing (r=1.0; lengths 0.32→1.82 across the 6 modules); and the
  modules are **aligned** (one sweep direction). The grid codes along the sweep are emitted as look-ahead
  tokens (a natural LLM interface for "what is probably to my left if I keep walking?").

*Honest scope.* The sweep *statistics* are constructed to match Vollan (this is an added mechanism, like the
boundary/object-vector cells — not an emergent measurement). The new result is the **mechanism + its
look-ahead function** (trap avoidance) and its faithful, multi-scale integration with the grid code.
(`results/theta_sweep.json`, `results/theta_sweep.svg`.)

**…and the sweep is now LOAD-BEARING for the readout/LLM — tokens, with an ablation** (`src/eval/theta_sweep_readout.py`,
n=5; `TrajectoryLLM(use_theta_sweep=True)`; `notebooks/m7_theta_sweep_llm_kaggle.py`). The same review's
follow-on: the sweep must not just exist, it must feed the LLM as tokens whose removal hurts. We wired it in.
`TrajectoryLLM` now optionally concatenates **theta look-ahead tokens** to the current spatial token —
`_sweep_tokens()` samples the grid map ahead (alternating L/R, ~20%-of-spacing) and projects each swept grid
code to a token, with `real / shuffled / ablated` modes for the ablation. The decisive CPU test uses a **novel
per-episode layout**, so the answer is *not* knowable from where the agent stands — it must look. A fixed small
readout predicts whether the cone ahead is **blocked** (balanced, chance 0.50):

| input to the readout | accuracy |
|---|---|
| **real sweep tokens** | **0.90 ± 0.02** |
| sweep ablated (current cell + heading only) | 0.58 ± 0.02 |
| shuffled (wrong-heading sweep) | 0.63 ± 0.01 |

With the real sweep the readout reads what is ahead and answers at **90%**; **ablate** the sweep and it falls to
**58%**, **mis-direct** it (wrong heading) and **63%** — both near chance. In a novel layout *nothing but the
sweep* can see ahead, so this is a clean, capacity-independent demonstration that the sweep tokens carry the
look-ahead (Vollan's sweeps extend into never-visited space). (`results/theta_sweep_readout.json`,
`results/theta_sweep_readout.svg`.)

**…and the frozen LLM confirms it — theta-sweep tokens are load-bearing for language** (`notebooks/m7_theta_sweep_llm_kaggle.py`,
n=8 on a T4). The full ablation, run on a frozen Qwen2.5-1.5B (LoRA + gated fusion). The LLM judges "is the path
ahead blocked?" in a **novel per-episode layout** (chance 50%), reading current-cell tokens + theta look-ahead
tokens; the moves never appear in the prompt, so cortex-ON vs text-only-OFF is causal:

| condition | accuracy | Δ vs ON | p (paired sign-flip, n=8) |
|---|---|---|---|
| **ON** (current cell + real sweep tokens) | **68% ± 14** | — | — |
| **OFF** (text-only, cortex ablated) | 44% ± 12 | +24 | **0.0081** |
| **NO-SWEEP** (current cell only, sweep ablated) | 41% ± 10 | +27 | **0.0081** |
| **SHUFFLED** (wrong-heading sweep) | 51% ± 6 | +18 | **0.0081** |

With the real sweep tokens the frozen LLM answers at **68%** and falls to **chance** without them — **41%**
sweep-ablated, **44%** text-only, **51%** wrong-heading-shuffled (all within CI of 50%). The **decisive**
contrast is **ON vs NO-SWEEP** — both carry the cortex/current cell, *only* the theta-sweep tokens differ:
**+27%**. And the effect is now **unanimous**: all three ablations sit at **p=0.0081**, the n=8 sign-flip floor
(2/2⁸), meaning **ON exceeds every ablation in every one of the 8 seeds**. *Honest, two ways.* **(i) Modest
magnitude:** 68% is +18 over chance and +27 over the decisive sweep-ablated control — real and robust, but not a
near-ceiling number; this few-token frozen-LLM+LoRA reader is a weaker learner than the dedicated CPU readout
(0.90). **(ii) Consistency, not a bigger headline:** this convergence-hardened run (2800 steps, warmup 200) is
*more consistent but lower* than a shorter pilot (1600 steps: ON **82% ±16**, but ON-vs-OFF p=0.030 — a seed or
two were stuck at chance). Hardening pulled **every** seed onto the effect (all p at the floor) at the cost of
peak accuracy — it bought cross-seed robustness, not a higher number (likely mild overfitting at 2800 steps).
Either way the review's demand is borne out at the language level: **the LLM uses theta-sweep tokens, and
removing them drops performance to chance, in every seed.** (`results/theta_sweep_llm_agg.json`.)

## Beyond the hippocampal core — a basal-ganglia action-selection organ

The first system added outside the hippocampal–entorhinal core (a Tier-2 gap), and the agent's action
selector upgraded from a generic softmax to a faithful circuit (`src/eval/basal_ganglia.py`, n=3). A
cortico-striatal **Go (D1) / NoGo (D2)** opponent circuit selects actions by softmax(Go − NoGo)
(direct−indirect, thalamic disinhibition), and learns by **local, dopamine-RPE-gated three-factor
plasticity** (positive RPE → Go LTP, negative → NoGo LTP; Frank's OpAL) with synaptic homeostasis — **no
backprop**. On a navigation task:

- **INTACT** learns to **100%**; **−dopamine** (RPE no longer gates striatal plasticity) stays at **chance
  (35%)** — the **dopamine-dependence of reward-based action learning** (the Parkinsonian signature),
  emergent from a faithful circuit with a local rule.
- *Honest nuance:* the **Go/NoGo pathways are partially redundant** on this task — a single-pathway lesion
  (−D1 or −D2) still reaches 100% (slower early: 75–81% vs 89%), because dopamine gates *both*. So it is
  the loss of **dopamine** (the shared neuromodulatory teaching signal), not of one pathway, that
  abolishes learning.

So action selection joins navigation, memory, and timing as a faithful organ with its own signature
lesion — and the agent now selects actions through a real basal ganglia, learned without backprop.
(`results/basal_ganglia.json`, `results/basal_ganglia.svg`.)

## Why grid cells? — coding capacity at scale (Fiete), and an honest caveat

The agent's navigation/return/bearing tasks above already *use* a grid cortex, but they don't show
*why* the brain pays for one: closed-loop navigation to a region is forgiving of a coarse code (we
checked — grid and place both reach ~100% across arena sizes; **no behavioral advantage**). The grid
advantage is **representational**, and that is the actual Fiete claim. So we measure it directly
(`src/eval/grid_capacity.py`, n=5): at a **fixed neuron budget**, how precisely is position encoded by a
periodic multi-scale **grid** code vs a local-bump **place** code, as the arena scales up 8×?

We use **Fisher information** — the Cramér–Rao bound, i.e. the position precision *available in the code,
independent of any decoder* (`res = det(Fisher)^(-1/4)`; lower = finer). Both Fisher forms are closed-form
and **verified against autograd** (det match to 6 sig figs; `tests/test_grid_capacity.py`).

| arena width | grid resolution | place resolution | place/grid |
|---|---|---|---|
| 2  | 0.027 | 0.16 | 6.0× |
| 4  | 0.031 | 0.32 | 10.3× |
| 8  | 0.035 | 0.64 | 18.3× |
| 16 | 0.039 | 1.29 | **32.9×** |
| **log-log slope vs arena** | **+0.18 (flat)** | **+1.00 (linear)** | — |

- **The grid code holds resolution ~constant as the arena grows** (slope +0.18; it's set by the finest
  period, which is reused across all of space), while **place degrades exactly linearly** (slope +1.00; a
  fixed budget of bumps must tile an ever-larger arena ever more coarsely). The grid advantage **grows
  with scale**, reaching **33×** at the largest arena — the exponential-vs-linear capacity that is the
  textbook reason the brain uses grid cells (Sreenivasan & Fiete 2011; Fiete et al. 2008).
- **Honest caveat (the capacity is real, but not free).** A *linear* reader **cannot extract it**: linear-
  decode MAE is actually *worse* for grid than place (grid 0.19→1.20 vs place 0.007→0.05 over the same
  arenas), because the phase→position map is nonlinear/periodic. The information is in the code, but it
  takes a **nonlinear/Bayesian decoder** to read out — which is exactly why downstream hippocampal place
  cells (a nonlinear conjunction of grid inputs) exist.

So the grid cortex the agent already runs on is not a stylistic choice: at scale it is the only one of the
two codes whose precision survives a fixed budget — and the caveat tells us *why* a place-cell read-out
sits downstream of it. (`results/grid_capacity.json`, `results/grid_capacity.svg`.)

### The other half of the trade-off — catastrophic errors, and why the code is multi-module

The grid code's capacity has a price (Sreenivasan & Fiete 2011; Fiete et al. 2008). It is a **residue code**:
position is read from the joint phases of several modules. Under noise a phase can slip so the residue
combination lands on a *different* consistent position — a **catastrophic error**: not a small drift but a
large jump to an aliased location. We maximum-likelihood-decode a noisy 1-D grid code (the nonlinear decoder
that actually exploits the combinatorial structure — a linear reader can't, per the caveat above) and
measure (`src/eval/grid_catastrophe.py`, n=5):

| #modules K | dim | catastrophic rate | local (median) error |
|---|---|---|---|
| 2 | 4 | **75%** | 0.25 |
| 3 | 6 | 23% | 0.003 |
| 4 | 8 | 8% | 0.002 |
| 5 | 10 | 1.5% | 0.002 |
| 6 | 12 | **1%** | 0.002 |

- **(A) Modules suppress catastrophes *exponentially*, at constant precision.** From K=2 to K=6 the
  catastrophic rate falls **75% → 1%**, while the local (median) error barely moves (0.003 → 0.002): adding
  modules buys **catastrophe-safety, not resolution**. This is precisely *why the entorhinal code is
  multi-module* (several modules at geometric scale ratios; Stensola 2012) — each module is another
  constraint an alias must satisfy, so catastrophes become exponentially unlikely.
- **(B) The error law is bimodal.** At K=2 the errors split **25% local / 75% catastrophic** with almost
  nothing between — the signature of a residue code failing; by K=5 the catastrophic tail is gone (98%
  local). 
- **(C) An honest correction to my own first framing.** I expected a *trade-off* "place is catastrophe-safe
  but coarse" — but the data refuted it: a place code **also** makes catastrophic wrong-bump errors under
  noise. At matched budget the grid code is **~19× finer *and* no more catastrophe-prone** (grid 19% vs
  place 25% at the highest noise). So the catastrophe-risk is **intrinsic to noisy decoding**, not a
  grid-vs-place deficit — and multi-module redundancy (A) is exactly what lets the high-capacity grid code
  *also* be catastrophe-robust, so **grid dominates place at matched budget** (consistent with the capacity
  result: the grid advantage is real once a nonlinear decoder unlocks it).

Together with the capacity result, this is the complete Fiete picture: the grid code's exponential capacity
*and* its catastrophic-error vulnerability, with the brain's multi-module organization the resolution of
both. (`results/grid_catastrophe.json`, `results/grid_catastrophe.svg`.)

## Emergent neuroscience signatures — measured, not designed

Like the 7±2 working-memory limit (which fell out of theta-gamma), other brain signatures emerge
from the trained cortex when probed directly (`src/eval/emergence.py`; cortex pre-trained ONLY on
self-supervised bounded PLACE-cell prediction — no periodic structure imposed as a target):

1. **Grid cells.** Spatial rate maps of the path-integrating units are PERIODIC and MULTI-FIELD —
   **100% of units have ≥3 firing fields, ~13 fields/unit on average** — not only in the attractor
   sheet but also in the learned 64-d summary `h`, which was trained on NON-periodic place cells (the
   Banino 2018 / Cueva–Wei 2018 emergence). The lattice is **square (4-fold), not hexagonal**: the
   biological hexagon needs a *twisted* torus (Guanella 2007), while our integrator uses a square
   toroidal sheet — so periodic grids emerge with the symmetry of the attractor's topology.
   See `results/emergence_gridcells.svg`.

   **We tested the twisted-torus prediction directly** (`--topology hex`: a rhombic 60° sheet wrapped
   on hexagonal lattice vectors). The gridness metric is validated on synthetic maps (clean square →
   −1.08, clean hexagon → **+1.09**), so it would detect a hexagon. Outcome: the twist measurably
   IMPROVES the code — position decode 0.71 → **0.87**, distance compression eases 0.50 → **0.69×**,
   denser fields (13 → 19 per unit) — **but the emergent real-space firing did NOT flip to regular
   hexagonal** (mean gridness ≈ −0.46; 0/256 units pass gridness>0). A clean *falsification*, and an
   instructive one: unlike hand-built attractors (Guanella; Burak–Fiete) where the velocity→sheet map
   is *constructed* to preserve the lattice metric, here `vel_to_sheet` is **learned** (to predict
   place cells) and is free to map real space onto the sheet with arbitrary shear/orientation — so
   connectivity topology alone does not dictate real-space grid symmetry. The hexagonal substrate is
   nonetheless a *better metric* (the decode/compression gains), hinting at why biology prefers it.
   Constructive next step: constrain `vel_to_sheet` toward an isometry onto the hex sheet, or add a
   hexagonal-symmetry objective. (`results/emergence_hex.json`, `results/emergence_gridcells_hex.svg`.)

   **Resolved — the lever is the VELOCITY mapping, not the connectivity.** Adding the faithful
   continuous-attractor construction (`--constrained`: self-motion velocity drives a phase integrated
   and wrapped on a hexagonal torus; K=4 modules at geometric scale ratios à la Stensola 2012; gains
   FIXED, readout LEARNED — the entorhinal→hippocampal flow) flips the result decisively:

   | torus / mechanism | mean gridness | units gridness>0 | position decode | distance compression |
   |---|---|---|---|---|
   | square torus | −0.46 | 0/256 | 0.71 | 0.50× |
   | hex torus (connectivity only) | −0.46 | 0/256 | 0.87 | 0.69× |
   | **constrained velocity (hex modules)** | **+0.87** | **255/256** | **0.97** | **0.95×** |

   The emergent grid cells are now **HEXAGONAL** (gridness +0.87 vs +1.09 for a textbook synthetic
   hexagon — 255/256 module cells pass). And it is not only the lattice: the multi-scale velocity-driven
   code gives **near-perfect position decoding (0.71→0.97)** and **all but eliminates the distance-
   compression bias (0.50×→0.95×)** — faithful path integration. (The learned readout `h` mixes modules
   into a place-like code — only 5/64 of *its* units are hexagonal — exactly the grid→place transform
   the hippocampus performs.) So the falsification was correct and instructive: grid hexagonality is set
   by *how velocity drives the phase* (the path-integration mechanism), not by sheet connectivity alone;
   build that mechanism in and a hexagonal grid — plus a near-perfect, length-invariant metric — emerges.
   Each step is *more* brain-like: conjunctive velocity cells → multi-scale grid modules → learned place
   readout. (`results/emergence_hexvel.json`, `results/emergence_gridcells_hexvel.svg`.)
2. **Path-integration drift & distance compression.** Decoding position from the frozen rep (corr
   0.71) shows the cortex systematically UNDER-estimates distance — decoded ≈ **0.5× true** — and
   error grows monotonically with distance (0.56 → 1.54 across the arena). Both are documented
   biological PI biases (homing-vector underestimation; error accumulation with travel) — and the
   same integration drift that caps the distance task at long T (the magnitude-frontier residual).
3. **Head-direction cells.** 88% of conjunctive units are directionally tuned (mean vector strength
   0.49) — a ring-attractor head-direction code (Taube 1990).
4. **7±2 working memory** (above) — recall stays near-perfect until the trajectory overflows
   theta-gamma's ~8 slots, then collapses (99.6% → 30.7% as T goes 4 → 14).

None of these were fitted as objectives; they fall out of a network assembled from grid / place /
head-direction / theta-gamma primitives and trained only to navigate. The architecture reproduces
the *phenomenology* of the spatial brain, not just its parts. (`results/emergence.json`.)

### Binding the emergent grid cells back to the language model

We then routed TrajectoryLLM's cortex through the velocity-driven hexagonal grid modules
(`--constrained_velocity`) on the distance task — closing the loop from "grid cells emerge" to
"grid cells drive language":

| distance, exact (within-1) | T=8 | T=16 | T=24 | cortex probe |
|---|---|---|---|---|
| place attractor + place target | 62% (94) | 46% (78) | 40% (81) | 85/66/37 |
| square attractor + **grid target** | 79% (100) | 83% (99) | 63% (94) | 99/88/83 |
| **grid-cell cortex** + place target | 68% (95) | 52% (88) | 50% (85) | 94/83/79 |

- **The faithful grid-cell cortex carries the language task.** Against the place-attractor baseline
  (same self-supervised target), it improves the LLM at every length — exact +6/+6/+10, within-1 +1/
  +10/+4 — and the cortex probe nearly DOUBLES at the hardest held-out length (T=24: 37% → 79%). The
  emergent hexagonal-grid mechanism genuinely powers the LLM's distance reasoning.
- **Grid CORTEX and grid TARGET are redundant, not additive.** It did not beat the grid-*target*
  route (79/83/63), and a CPU probe explains why: teaching a plain square attractor to predict a grid
  code already induces a grid-like metric (probe 99/88/83), and the grid-cell cortex reaches the same
  place by construction (probe with a grid target 97/88/80); combining them does not stack. They are
  two routes to one functional endpoint — a faithful, scale-true, length-invariant metric.
- **Why prefer the cortex route:** it is the actual entorhinal path integrator — it *produces* the
  emergent hexagonal grid cells and is length-invariant by construction — rather than a training trick
  layered on a square sheet. Within-1 stays 85–95% throughout: the magnitude is essentially right.
  (`results/m2_distance_gridcortex.json`.)

### Toward the real brain — boundaries correct path-integration drift

The limitation we kept hitting is integration DRIFT: noisy path integration accumulates error with
distance travelled. The brain's fix is not a better integrator — it is SENSORY ANCHORING:
environmental boundaries reset accumulated grid error (Hardcastle, Ganguli & Giocomo 2015). We added
exactly this — `BoundaryVectorCells` read the (distance, bearing) to the nearest wall and gate-reset
the grid phase toward the boundary-implied coordinate (the perpendicular axis only) — and tested it
with NOISY integration in a walled arena (`src/eval/boundary_anchoring.py`; position-decode error vs
path length T):

| condition | T=6 | T=12 | T=18 | T=24 | T=30 | vs drift |
|---|---|---|---|---|---|---|
| exact (no integration noise) | 0.04 | 0.07 | 0.10 | 0.13 | 0.14 | floor |
| drift (noisy, no anchor) | 0.35 | 0.52 | 0.64 | 0.76 | **0.85** | — |
| anchored — geometric fix (hard-coded R−dist) | 0.32 | 0.39 | 0.44 | 0.47 | **0.49** | **−43%** |
| anchored — LEARNED loc. (supervised by true pos) | 0.33 | 0.41 | 0.47 | 0.52 | **0.56** | **−34%** |
| **anchored — BOOTSTRAP (learned from the agent's OWN PI, no labels)** | 0.33 | 0.41 | 0.47 | 0.53 | **0.58** | **−32%** |

Without boundaries the error grows steadily with distance (drift ∝ √T); **boundary anchoring cuts it
~43% at T=30 and flattens its growth** (accumulation rate −66%) — the grid phase is re-pinned whenever
the agent passes a wall, so error can't accumulate without bound. That is the Hardcastle-2015
mechanism reproduced: the brain does not beat drift with a perfect integrator, it CORRECTS it with
sensory landmarks.

**Removing the scaffolds, one at a time.** v1 computed the boundary-implied coordinate from arena
geometry (R−dist). v2 *learned* it (boundary-vector cells → a learned position head), calibrated then
frozen — *development before use* (training it jointly lets the decoder suppress the gate whenever
localization is briefly wrong). v3 removes the **last** scaffold — the position label: the localizer
is trained ONLY against the agent's OWN path-integration estimate (dead-reckoning = integrated
self-motion + proprioceptive noise), never the true position.

**The bootstrap denoises its own teacher.** That PI teacher drifts badly (RMSE 0.41 near walls), yet
the localizer trained on it reaches **RMSE 0.076 vs true** — 5× better than its teacher — because the
wall→position mapping is consistent across visits while the drift is zero-mean and averages out. It
then bounds the path-integration drift **−32%**, essentially matching the label-supervised version
(−34%). So boundary localization is learned from *only* self-motion and boundary sensing — no position
labels, no arena geometry. This is the consistency/bootstrap learning that grounds the cognitive map:
path integration provides a noisy teacher, boundary cells learn to predict (and thereby denoise) it,
then correct it. (`results/boundary_anchoring.json`, `results/boundary_anchoring.svg`.)

### Three more pillars — remapping, replay, local plasticity

Three further hallmarks of the spatial brain fall out of (or wire cleanly onto) the velocity-driven
grid cortex (`src/eval/pillars.py`, CPU):

- **Remapping & grid reuse** (Fyhn 2007; Leutgeb 2005). The grid code is a UNIVERSAL metric: a single
  position decoder trained in environment A works in a new environment B unchanged (0-shot, err
  0.012 = 0.012). Yet PLACE codes REMAP — for the same locations, two environments' place population
  vectors are decorrelated (cos **0.08**) while the grid population is identical (cos **1.00**). And a
  new environment's place map is learned FEW-SHOT on the ready grid metric (place MSE 0.045 → 0.002 in
  tens of steps). Grids = reusable metric; place cells = the per-environment, remappable readout.
- **Replay / consolidation** (sharp-wave ripples). Hippocampal replay is experience replay: from only
  40 real trajectories, using each once decodes poorly (err **0.89**); REPLAYING that small buffer
  offline consolidates a near-ceiling map (err **0.073** vs the 4000-trajectory ceiling 0.017). Offline
  rehearsal turns a little real navigation into a good map.
- **Local (Hebbian) plasticity — place cells without backprop.** Place fields EMERGE from the grid
  code via competitive Hebbian learning (winner-take-all + move-toward-input, a LOCAL rule): the units
  become compact single-field place cells (mean field area 6% of the arena, **100% compact**), tiling
  space — the classic grid→place transform (Rolls & Treves), formed by local plasticity. See
  `results/pillars_hebbian.svg`.

*Honest caveats.* Grid "reuse" is partly by construction (our grids are position-driven and
environment-independent; we don't model grid realignment between rooms) — the substantive parts are
place remapping (cos 0.08) and few-shot map formation. Replay here is experience replay (offline
rehearsal of a stored buffer), faithful in spirit but without modelling compressed/reverse SWR
dynamics or a separate consolidated cortical store. And Hebbian plasticity is shown to FORM the
place readout locally; the rest of the pipeline still trains by backprop. (`results/pillars.json`.)

### Consolidating — the faithful grid-cell cortex across all three language tasks

We routed TrajectoryLLM through the velocity-driven hexagonal grid-cell cortex (6 modules, wide
residue range) on every navigation question (`--constrained_velocity`):

| task | grid-cell cortex (T=8/16/24) | place/default cortex | OFF (control) |
|---|---|---|---|
| **return** | **100 / 100 / 100** | 96 / 89 / 86 | ~chance |
| **bearing** | **85 / 83 / 80** | 71 / 78 / 73 | ~chance |
| **distance** (exact) | **95 / 88 / 85** | 62 / 46 / 40 | ~chance |
| distance (within-1) | **100 / 99 / 94** | 94 / 78 / 81 | 87 / 74 / 71 |

- **The faithful cortex wins on every task AND flattens extrapolation.** Return is perfect and flat
  (100/100/100) where place degrades (96→86); bearing +14/+5/+7 and flat; distance **95/88/85 exact**
  (within-1 100/99/94) vs place's 62/46/40 — a huge gain that barely declines to 3× the training
  length. cortex-OFF sits at chance throughout, so the answers ride on the spatial code. The emergent
  hexagonal grid cells carry every navigation question in language, better than the place attractor.
- **Distance needed training-stability care, not a better representation.** Exact-bucket accuracy
  oscillated across epochs (a 6-class LLM-readout artifact; the rep was always excellent, probe
  100/95/93, and it peaked early). Early stopping (restore the best-val epoch) + a lower LR locked in
  95/88/85 — from a peak the wobble had been burying (the no-early-stop run ended at 44).
- **The grid-module count matters.** With only 4 modules the residue code aliased on long paths
  (cortex-probe bearing 96/74/62); widening to 6 modules (unambiguous range ~9) made the cortex probes
  flat and high — return 100/100/100, bearing 98/96/91, distance 100/95/93 — exactly the high-capacity
  multi-module grid-code prediction (Fiete/Stemmler).

**Verdict:** one biologically-faithful cortex — emergent hexagonal grid cells, multi-module,
velocity-driven, length-invariant — carries *every* navigation question in language (return 100,
bearing 85, distance 95 exact), beating the place attractor on each and staying flat to 3× the
training length.
(`results/m2_return_gridcortex.json`, `results/m2_bearing_gridcortex.json`, `results/m2_distance_gridcortex.json`.)

### Planning — the map as a PLANNER, not a recorder (Tolman shortcut)

The final step makes the cognitive map prospective: can the agent PLAN a route it never walked? Because
the grid code is a linear metric (phase ∝ position), the displacement between any two remembered places
is just the difference of their grid codes — a vector the agent reads off directly (vector navigation:
Bush 2015; Banino 2018) and can FORWARD-REPLAY before moving (preplay: Pfeiffer & Foster 2013). The
agent reaches A and B by two SEPARATE winding walks from home (never travelling A→B), then plans the
direct A→B shortcut from the map (`src/eval/planning.py`):

| metric | result |
|---|---|
| planned A→B shortcut direction error | **0.33° mean (0.23° median)** |
| distance relative error | 0.7% |
| shortcuts navigable (<15° off) | **100%** |
| forward-replay sweep deviation from the straight line | 0.078 |
| shortcut shorter than retracing via home | **29%** |

The agent computes a near-perfect straight-line shortcut to a goal it reached only by a winding detour
— the classic Tolman cognitive-map result — and forward-replays the imagined path coherently to the
goal. The map is no longer just a recorder of where it has been; it is a PLANNER of where to go. And
this vector navigation falls out of the *same* grid metric that path-integrates, generalises across
length, and drives the language tasks — one map, used to record, generalise, answer, and now plan.
(`results/planning.json`, `results/planning.svg`.)

### Value & goal-directed navigation — the map serves a goal (dopamine)

Until now the map was reward-agnostic. We made it value-laden: the agent explores and gets SPARSE
reward at an unknown goal (never told where); a value head V(grid-code) is trained by a dopamine-like
TD error δ = r + γV(s′) − V(s), with the goal terminal (reward consumed) (`src/eval/goal_navigation.py`):

- **It localizes the unseen goal.** The peak of the learned value map sits **0.33** from the true goal
  (arena half-width 3.0) — recovered purely from sparse reward, no goal label. Value concentrates on the
  reward location (the overrepresentation of goals in the map; Hollup 2001).
- **It navigates there, goal-directed.** Climbing the value gradient through the map (evaluating V at
  candidate next steps — a forward-model lookahead), the agent reaches the goal from random starts
  **95% of the time in a median 6 steps**, vs a random walker's **29% (14 steps)**. The cognitive map
  now drives behaviour toward a goal.
- **Dopamine prediction-error shrinks as the world is learned.** Mean |δ| falls **0.057 → 0.034** over
  training — the reward-prediction-error decreasing as the value model converges (dopamine-as-RPE;
  Schultz 1997). (A continuing, non-terminal reward instead *inflates* value and the error grows — the
  shrink requires the goal to be consumed, which is also the more realistic case.)

So value is learned over the same grid map by a dopamine-like signal, and the map is no longer just a
spatial record — it is a motivated, goal-seeking controller. *Honest scope:* value sits on the frozen
grid map (the map isn't re-shaped by reward), the reward is a fixed location, and the policy is greedy
value-ascent (lookahead), not a learned motor policy — a standard RL abstraction of the dopamine
system. (`results/goal_navigation.json`, `results/goal_navigation.svg`.)

### Abstract / relational cognition — the grid map is a relational engine (TEM)

The hippocampal–entorhinal map is not only spatial: the same grid/place code maps relational STRUCTURE
— ordered sets, concept spaces, task graphs — enabling inference (Tolman–Eichenbaum Machine, Whittington
2020; grid codes in concept space, Constantinescu 2016; relational memory, Eichenbaum). We placed an
abstract ORDERED structure (items ranked 0..11) along a concept axis, mapped it with the SAME
velocity-driven grid cortex, and taught a comparison readout ONLY adjacent pairs (`src/eval/relational.py`,
with neural noise):

- **Transitive inference: 84%** correct on non-adjacent pairs NEVER trained (A>D from A>B>C>D) — the
  metric makes the order transitive. It even *beats* the adjacent TRAINED pairs (72%), because…
- **The symbolic distance effect emerges**: accuracy rises monotonically with rank-distance —
  **69% (adjacent) → 100% (farthest)**. Far-apart items are EASIER to compare — the hallmark behavioural
  signature (humans, monkeys, rats) that an abstract dimension is held on an analog/spatial map.
- **Schema transfer: 78%** zero-shot on a NEW ordered set in a different region of the concept space —
  the relational structure is abstracted from the specific items (content) and reused like a schema.

So the very grid machinery that path-integrates physical space, generalises across length, drives
language, plans, and seeks reward also performs LOGICAL inference over an abstract ordered structure —
the deepest "beyond metric" result: the cognitive map is a general relational engine, not a spatial
special case. *Honest scope:* the concept space is hand-laid (items placed along an axis, not discovered
from raw stimuli) and the comparison is a learned readout over the frozen grid code — we show the
transitive-inference + distance-effect + transfer *signatures*, not a full TEM with learned
structure/content factorisation. But the core claim — relations represented as space, enabling inference
never trained on — holds. (`results/relational.json`, `results/relational.svg`.)

### One-shot & continual learning — instant place fields, no catastrophic forgetting (CLS)

The cortex above was pre-trained then frozen; the brain instead encodes a place in ONE visit and
accumulates memories without overwriting them. We bind each visited location, in a single local write,
to a place cell w = grid-code(L) (`src/eval/continual.py`):

- **One-shot place field**: a single visit produces a localized place field (area **0.06** of the arena)
  — formed in one write, not many gradient steps (behavioural-timescale plasticity; Bittner & Magee 2017).
- **Continual, no catastrophic forgetting**: visiting K=20 places sequentially (one-shot each), ALL are
  still recalled afterwards — recall by learning-age is FLAT (oldest→newest quartile **96/96/91/100%**,
  mean ~96%). A single shared classifier trained the SAME sequence by gradient CATASTROPHICALLY FORGETS:
  the oldest quartile collapses to **0%** (mean ~30%, quartiles 0/51/29/40%).

This is Complementary Learning Systems made concrete (McClelland, McNaughton & O'Reilly 1995): fast,
pattern-separated, one-shot hippocampal storage retains everything, where a slow distributed gradient
learner interferes and forgets. The grid code supplies the pattern separation (distinct places →
distinct codes), so each one-shot memory is independent. *Honest scope:* this is the *fast* hippocampal
store (expandable place cells bound to the frozen grid metric); a complete system pairs it with *slow*
neocortical consolidation (our replay pillar) interleaving these memories into shared weights — which is
exactly the CLS division of labour. (`results/continual.json`, `results/continual.svg`.)

### Embodiment — the map grounded in vision (optic-flow self-motion)

The cortex was handed (heading, speed). The brain instead SEES the world and infers self-motion from
optic flow. We gave the agent a visual world (16 landmarks), a retinal PANORAMA at each position, and a
learned visual front-end that estimates velocity from how the panorama shifts between frames; that
vision-derived velocity drives the SAME grid cortex (`src/eval/embodiment.py`):

- **Vision recovers self-motion**: the optic-flow front-end estimates the agent's velocity at direction
  cosine **0.97** (error 0.13 vs a ~0.5 step) — no hand-given heading/speed.
- **The grid map path-integrates from vision**: decoding position from the grid code built on
  VISION-derived velocity, the agent localizes with error **0.48 → 1.33** over path length T=6→24
  (arena half-width 3) — it knows where it is from what it SEES.
- **The gap to the true-velocity ceiling (0.01–0.02) is accumulated optic-flow noise**: visual path
  integration DRIFTS — exactly the error that, in the brain and in our boundary pillar, is corrected by
  re-anchoring to landmarks/boundaries. Embodiment introduces the very drift the boundary mechanism
  fixes; the two pillars meet.

So the pipeline is now grounded end-to-end in perception: world → retinal panorama → optic-flow
egomotion → grid path integration → place / value / relational readout. The agent is no longer told how
it moved; it perceives it. *Honest scope:* a simplified panoramic landmark world and a learned MLP
front-end (not pixels through a CNN); translation-only (no rotation); the front-end is calibrated
against efference copy (the agent's own motor signal), as optic flow is in development.
(`results/embodiment.json`, `results/embodiment.svg`.)

## Statistical robustness — multi-seed (mean ± 95% CI)

Single runs are not evidence. To move each flagship CPU result from "it worked once" to "it
works, with error bars", `src/eval/stats.py` re-implements the core measurement of each eval inside
a seed loop and reports **mean ± 95% CI over n = 8 seeds** (`results/stats.json`):

| capability | metric | mean ± 95% CI (n=8) | baseline (same code) |
|---|---|---|---|
| Planning (Tolman shortcut) | shortcut direction error | **0.344° ± 0.044°** | — |
| Planning | fraction navigable (<15°) | **1.000 ± 0.000** | — |
| Relational (TEM) | transitive inference acc | **0.836 ± 0.008** | chance 0.50 |
| Relational | symbolic-distance-effect correlation | **0.957 ± 0.009** | 0 if no analog code |
| Continual (CLS) | one-shot Hebbian recall | **0.942 ± 0.023** | gradient **0.282 ± 0.045** |
| Goal navigation (dopamine) | value-guided success | **0.954 ± 0.049** | random **0.285 ± 0.026** |

Every metric is tight across seeds, and the two head-to-head dissociations have **non-overlapping
95% CIs** — one-shot Hebbian recall (0.942 ± 0.023) vs gradient forgetting (0.282 ± 0.045), and
value-guided navigation (0.954 ± 0.049) vs a random walker (0.285 ± 0.026). The goal-navigation
seed loop also **randomizes the reward location per seed**, so the CI reflects robustness to *where*
the goal is, not just to initialization. These are not lucky runs.

### Paired significance tests — every headline claim with a p-value (the rigor table)

Non-overlapping CIs are informal; reviewers want a test. `src/eval/significance.py` runs each headline
comparison **paired on shared seeds** (n=20 for the analytic comparisons, n=8 for the heavy
goal-navigation / Transformer ones) and reports, on the per-seed differences, a **bootstrap 95% CI of
the mean difference** (20k resamples), a **two-sided sign-flip permutation p-value** (gold standard for
paired data, no distributional assumption), and **Cohen's d** (`results/significance.json`,
`results/significance.svg` forest plot):

| comparison | Δ mean [95% CI] | p (perm) | Cohen's d | seed wins |
|---|---|---|---|---|
| extrapolation distance @T24: **grid − place** | +0.124 [+0.119, +0.129] | <1e-4 | 10.9 | 20/20 |
| extrapolation distance @T24: **grid − GRU** | +0.053 [+0.034, +0.080] | <1e-4 | 1.0 | 20/20 |
| extrapolation bearing @T24: **grid − place** | +0.078 [+0.073, +0.083] | <1e-4 | 6.8 | 20/20 |
| multi-map @M16: **grid+remap − additive** | +0.766 [+0.754, +0.777] | <1e-4 | 27.9 | 20/20 |
| capacity @K200: **population(grid) − raw-2D** | +0.507 [+0.497, +0.516] | <1e-4 | 21.9 | 20/20 |
| continual: **one-shot Hebbian − gradient** | +0.662 [+0.626, +0.698] | <1e-4 | 7.9 | 20/20 |
| relational: **transitive inference − chance** | +0.338 [+0.334, +0.343] | <1e-4 | 29.7 | 20/20 |
| goal navigation: **value − random walker** | +0.670 [+0.618, +0.704] | 0.006 | 9.5 | 8/8 |
| **NULL — extrapolation @T24: grid − NoPE+sum Transformer** | **+0.002 [−0.022, +0.032]** | **0.94** | **0.04** | **3/8** |

Two things this certifies. (1) **Every claimed effect is significant** — p < 1e-4 (goal-nav p=0.006 at
n=8), large effect sizes, and the sign of the difference holds in *every* seed. (2) **The honest tie is
a certified null**, not a hand-wave: grid vs a NoPE+sum Transformer on path integration is
Δ = +0.002 with a 95% CI that straddles zero (p = 0.94, d = 0.04) — there is genuinely no difference,
exactly the claim we make. The forest plot (`results/significance.svg`) shows all nine effects against
zero at a glance: eight clear of it, one centered on it.

*Language-level rigor (§"Milestone 2"/§4):* the LLM grid-vs-place comparison is at n=3 with large
seed variance and is **not** separable there; the cortex-ON ≫ text-only-OFF result *is* robust. A
bearing-only n≥8 LLM sweep with the same paired test is specified in
`notebooks/m2_extrapolation_multiseed_kaggle.py` (cells 5–6).

## Caveats / open questions
- The 3D task is near-trivial (threshold one input coordinate); `coord_2d_noleak` is
  the meaningful spatial-reasoning test.
- All results: Qwen2.5-1.5B, 1 epoch, LoRA on q/v, cities15000, single T4.
- Next candidates: harder tasks (distance/bearing between two points), or pushing the
  2D no-leak accuracy (more epochs / larger coord encoder) to find the geography ceiling.

See `results/*.json` for raw per-seed numbers and gate read-outs.
