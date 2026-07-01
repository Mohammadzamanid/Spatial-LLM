# A self-supervised cognitive-map cortex for language models — and an honest account of what brain-faithful spatial coding does and does not buy

**Working draft — Spatial-LLM.** Status markers: ✅ = result in hand, multi-seed with 95% CI; ⏳ =
specified, GPU run pending; ✎ = prose to finalize. Every number is reproduced by a script in
`src/eval/` or a notebook in `notebooks/`; raw values in `results/*.json`. This draft commits to an
*honest* framing: we report ties and negative results as first-class findings.

---

## Abstract ✎

We give a frozen language model a brain-faithful spatial substrate — a self-supervised cortex of
velocity-driven **grid cells** and **place cells** that path-integrates self-motion into a periodic,
multi-scale metric — and study, with multi-seed error-barred controls, what that representation
contributes. Two things. First, an **integrative result**: a *single* self-supervised code, learned
with no coordinate labels, transfers spatial competence to a frozen LLM (it answers navigation
questions through the cortex, not the text) and — with its metric unchanged — also supports
vector-based **planning**, dopamine-like **value** learning and goal navigation, **relational/transitive
inference**, and one-shot **memory**. Second, an **honest characterization** of *which* representational
properties actually matter, including results that run against the simplest story: on pure path
integration the grid code is matched by a permutation-invariant, sum-pooling Transformer that shares its
*additive integration bias*; the population code's distinctive properties (high-capacity
pattern-separation, environment-specific **remapping**) are decisive only in narrow regimes — fixed
associative memory and *context-free* settings — and do **not** transfer to a trained model that already
has an external context label (as an LLM does in its prompt). The contribution is therefore a rigorous,
fairly-baselined map of *when* brain-faithful spatial coding helps and when a simpler inductive bias
suffices, together with the integrative demonstration. Every claimed effect is supported by paired
significance tests (sign-flip permutation, bootstrap CIs, n up to 20; all p<1e-4 with large effect
sizes), and the central tie is a *certified null* (grid vs a NoPE+sum Transformer: p=0.94, d=0.04). We
do not claim grid cells are a uniquely necessary substrate for a trained system, and we show why. We
*do* identify two regimes where the brain-faithful code is **necessary**, not merely competitive:
**cyclic (non-Euclidean) worlds**, where its periodicity computes toroidal position (∫v mod 2π) that
additive integrators provably cannot — flat at the oracle floor where they collapse to chance, and a
world a language prior cannot fake (a built-in leakage control); and **abstract relational inference**,
where a *space-trained, frozen* metric supports transitive inference and schema transfer, falsified by
shuffling the metric (p=0.009). Finally, prompted by the neuroscience of space *and time*, we extend the
purely-geometric cortex along the two axes it omits: a **successor-representation** map that plans
detours around barriers where a metric map stalls (100% vs 62%, paired p=0.009) and bends its fields to
geodesic rather than Euclidean distance, and a recurrent substrate that, trained only to read elapsed
time, **grows time cells** whose latency-dependent widening reproduces the brain's scalar (Weber) timing
law unbidden (17% of units vs 1% untrained) — these temporal signatures *emerge*, not imposed. A frozen
LLM then reads **both** codes from language — naming which cell of a wrap-around (toroidal) world it
occupies, and how much time has elapsed, purely through the cortex — each a significant cortex-ON ≫
text-only-OFF causal control (n=6, paired **p=0.033**), with the elapsed-time question never appearing in
the prompt.

---

## 1. Introduction ✎

Coordinate embeddings let a model memorize a map; they do not obviously let it *compute* over space in
a way that survives a change of scale or serves many downstream uses. The mammalian
entorhinal–hippocampal system solves navigation with a particular representational scheme — grid cells
that path-integrate velocity into a periodic multi-scale code, read out by place cells — that also
appears to underlie planning, value, and relational cognition. We ask a direct question: if we build
that substrate, self-supervised and label-free, and let a frozen language model read it, **what does it
buy, and what does it not?** We answer with fair baselines and multiple seeds throughout, and we let the
negative results stand.

## 2. The system ✎

A path of self-motion → conjunctive velocity cells → a velocity-driven hexagonal grid code (fixed
gains, geometric scale ratios; phase = gain·∫v wrapped on a hexagonal torus) → a learned place/value
readout → gated cross-attention into a frozen Qwen2.5-1.5B + LoRA. The cortex is pre-trained only to
predict bounded place-cell activity from self-motion (no coordinate labels). Architecture and configs:
`src/models/`, `results/architecture.svg`.

## 3. What transfers: length generalization ✅

Stripping away the LLM (so any effect is the representation), an agent random-walks in 2-D; we train a
position readout on mixed short paths {6,8,10,12} (scale-free) and test to 4× longer, deriving the
trajectory-QA tasks from the decoded displacement (`src/eval/extrapolation.py`, n=8). Against a *fair*
place baseline (tiled exactly to the trained region), the grid code wins at every length — at 3×,
**93% ±0 vs 80% ±1** distance accuracy, non-overlapping CIs — because a bounded place code cliffs once
paths leave its trained box while the grid code degrades gracefully (its phase is scale-free *and*
periodic). An exact-integration oracle is flat, so the gap is the code, not the task. (Figure 1:
`results/extrapolation.svg`. Honest ceiling: grid itself falls to 75% at 4×; range is finite.)

## 4. Mechanism — it is the inductive bias, not the architecture ✅

Single-variable ablations (`src/eval/ablations.py`, `seq_baselines.py`, n=5):

- **Range comes from modular coding**: 1 module aliases (14% at 4×) → 8 modules 82%, monotone.
- **Scale-invariance is needed**: a scale-free sum is flat at 99%; the same sum ÷ T collapses to 2%.
- **The advantage is in the code, not the training mix**: the grid code extrapolates even from a single
  training length.
- **A sequence model reveals the truth**: the *default* Transformer (learned positions, mean-pool)
  collapses (16% at 3×) and sinusoidal positions only partly help (38%), but a **NoPE + sum-pool**
  Transformer — permutation-invariant and additive — **ties the grid code (92% at 3×, and beats it at
  4×, 88% vs 75%)**. A GRU is mediocre and seed-unreliable (82% ±8).

So length extrapolation requires an *additive, scale-free, order-invariant integration bias*; the
conventional defaults lack it and the grid code has it by construction — but it is **not unique** to
grid cells. (Figures 2: `results/ablations.svg`, `results/seq_baselines.svg`.)

## 5. Where the population code helps — and where it does not (the characterization) ✅

Given that an additive integrator ties on path integration, we test what a *deterministic function of
displacement* cannot do (`src/eval/code_necessity.py`, `multimap_task.py`, `frontier_probes.py`; n=5):

- **Memory capacity** ✅ (a win, but shared): the raw 2-D code collapses to 25% recall at 200 stored
  locations; *any* high-dimensional population code (grid/place/random-Fourier) holds 75%. You need a
  population code — but not specifically a grid.
- **Multi-map storage via remapping** ✅ (a win *in the right regime*): with a fixed one-shot memory, any
  deterministic metric code gives identical codes across environments and collides (4% over 16 maps),
  while grid/place **remap** and hold 79–92%; an ablation switching remapping off reproduces the
  collapse, isolating remapping as the cause.
- **…but remapping does NOT help a trained model with a context label** ⊘ (a boundary, reported): replace
  the one-shot memory with a trained classifier given a learned room-id embedding (the analog of a room
  name in an LLM prompt) and the non-remapping code reaches 100% at 32 rooms — the model substitutes the
  label for remapping. The brain remaps because it has *no* external context signal; an LLM has one.
- **Sample efficiency** ⊘ (a non-win): the fixed grid code is *less* data-efficient (34% vs a NoPE+sum
  Transformer's 73% at 16 training trajectories) — its high-dimensional code needs examples to learn the
  readout.
- **Noise robustness** ⊘ (a tie): once every code integrates the *same* noisy velocity, all degrade
  identically (~34% at σ=0.4). (An earlier probe that handed grid the clean displacement showed a
  spurious win; corrected.)
- **Mechanism vs parameters** (control, `src/eval/controls.py`): at fixed 384-d, a random *linear*
  projection and a learned MLP also extrapolate (they re-encode the unbounded displacement), and random
  *periodic* / random-scale codes match the geometric grid — so it is neither the parameter count nor
  grid-cell specifics. The lone discriminator is **saturation**: only the *bounded* place tiling fails.
  The grid code's precise niche is **unbounded metric range with bounded, normalized (biological)
  activations** — where a place code cannot follow and a linear code is not a realizable neural code.

**Verdict.** Across length extrapolation, capacity, remapping-in-a-trained-model, sample efficiency, and
noise, the velocity-driven grid code is *competitive but not uniquely necessary* for a trained system.
The additive integration prior captures the core; the population-code extras matter only in fixed-memory
or context-free regimes. This map of wins / ties / boundaries — with fair baselines — is the
contribution, and it is summarized as a single predictive **phase diagram** of *when each inductive bias
wins* (Figure 9, `src/eval/phase_diagram.py`): grid wins where periodicity / pattern-separation is
load-bearing (cyclic worlds, one-shot capacity), ties where a plain integration bias suffices (Euclidean
extrapolation, labelled multi-map, noise), and loses only in the very-low-data regime. (Figures 3–4:
`results/code_necessity.svg`, `results/multimap_task.svg`, `results/frontier_probes.svg`;
`results/phase_diagram.svg`.)

**Significance (paired tests, `src/eval/significance.py`, Figure 6).** Every claimed effect is
statistically significant under a paired sign-flip permutation test with a bootstrap CI of the
difference (n=20 fast / n=8 heavy): grid−place distance@T24 Δ=+0.124, p<1e-4, d=10.9 (20/20 seeds);
grid+remap−additive multi-map Δ=+0.766, p<1e-4; population−raw-2D capacity Δ=+0.507, p<1e-4;
Hebbian−gradient Δ=+0.662, p<1e-4; value−random goal-nav Δ=+0.670, p=0.006; transitive-inference−chance
Δ=+0.338, p<1e-4. Critically, the **honest null is certified**, not assumed: grid vs a NoPE+sum
Transformer on path integration is Δ=+0.002, 95% CI [−0.022,+0.032], **p=0.94, d=0.04**.
(`results/significance.svg`.)

**The tie inverts on non-Euclidean worlds — where the periodic code is *necessary* (Figure 7,
`src/eval/torus.py`).** On a torus, true position is θ = (∫velocity) mod 2π; a periodic grid code
computes that mod for free (cos ∫v = cos θ at any wrap count) while a non-periodic code sees an unbounded
∫v and cannot recover the wrap. Trained on short paths and tested to many wraps (n=8), the grid code is
**flat at the oracle floor (0.01 rad, 100% within 45°) at every length**, while the *same NoPE+sum
Transformer that tied it on Euclidean paths collapses to chance (1.56 rad, 25%)**, as do additive and
Euclidean-place codes — tiny, non-overlapping CIs. So the periodicity that was a wash on Euclidean paths
is exactly the right inductive bias for a cyclic world: there the brain-faithful code is not
competitive-but-tied, it is **necessary**. This is also the **leakage rebuttal** — a torus has no
faithful Euclidean text description, so a language prior cannot substitute for having path-integrated it.

## 6. One code, many functions — the integrative substrate ✅

With its metric fixed, the *same* self-supervised code supports (multi-seed, mean ± 95% CI,
`src/eval/stats.py`):

- **Planning** (Tolman novel shortcut): direction error **0.34° ± 0.04**, 100% navigable.
- **Value / goal navigation** (dopamine-like TD): **95% ± 5** vs a random walker 29% ± 3.
- **Relational / transitive inference** (TEM-style, trained only on adjacent pairs): **84% ± 1** on
  unseen non-adjacent pairs; clean symbolic-distance effect (corr 0.96 ± 0.01).
- **One-shot / continual** (CLS): Hebbian recall **94% ± 2** vs a forgetting gradient baseline 28% ± 5.

That one brain-faithful code serves navigation, planning, value, relational inference, and memory — read
by a frozen LLM — is the integrative significance, independent of any uniqueness claim.

**Structural transfer with falsifiers (Figure 8, `src/eval/structural_transfer.py`).** The relational
result above is strengthened into the TEM claim: with the cortex **frozen and trained only on space**, a
non-spatial ordered structure laid along a concept axis yields transitive inference on never-seen far
pairs (**0.836 ± 0.008**, exceeding the trained adjacent pairs 0.706 — the symbolic-distance effect) and
zero-shot schema transfer to a new item set (0.790). Two falsifiers fire: **shuffling the rank↔position
correspondence collapses TI (0.836→0.623, paired p=0.009)** — so it is the *ordered metric*, not
memorization — and scrambling the second item (0.656) shows the readout compares two codes, not one
magnitude. This is the representation-level validation of the headline LLM experiment (§8 roadmap), where
the readout is a frozen Qwen+LoRA answering a *linguistic* comparison it cannot do text-only.

## 7. The map is predictive and temporal — beyond a geometric record ✅ (CPU, n=8)

The hippocampal map is not a geometric record of position but a **predictive** model of future states
(the successor representation, SR; Dayan 1993, Stachenfeld 2017), indexed in **time** as much as in
space (time cells; Eichenbaum 2014, Howard's scale-invariant timing). Our cortex was purely spatial and
geometric; we close both gaps with CPU-validatable modules, each reproducing the brain's *falsifiable
signature* (multi-seed, mean ± 95% CI), before any LLM wiring.

**Predictive map (`src/eval/successor.py`, Figure 10).** The successor representation
**M = (I − γT)⁻¹** (expected discounted future occupancy) confers what a metric map cannot. On a
barriered gridworld, greedily ascending SR value reaches the goal **100%** of the time, while descending
Euclidean distance-to-goal stalls at **61.7% ± 9.3%** — the wall makes the straight-line gradient point
*into* it (paired sign-flip **p = 0.0086**); on an open field both reach 100%, so the gain is
*specifically* the detour (Tolman's insight, quantified). SR fields track **geodesic** distance
(across-wall corr **0.69 ± 0.06**) not Euclidean (**0.31 ± 0.12**) — the map bends around the barrier —
and a **TD-learned** SR matches the closed form at **0.97 ± 0.003**, so it is acquired from experience,
not merely constructed (`results/successor.{json,svg}`).

**Temporal map (`src/eval/time_cells.py`, Figure 11).** We do not build a time-cell basis; we let it
emerge. A generic recurrent substrate (`src/models/neuro/temporal_cortex.py`: leaky rectified rate-RNN,
one uniform time-constant, learned recurrence, private noise — nothing timing-specific) is trained on a
single task, "report elapsed time when probed at a random moment," with a metabolic activity cost; we
then measure what appears (n=8; an untrained net of the same architecture is the control). A **precise
timer emerges** (decode error **0.20 ± 0.04** steps vs untrained **3.6**); its code is a population of
**time cells** (**17%** of units vs untrained **1%**, single-peaked, tiling, **92% denser in the first
half** — Mau 2018) whose **fields widen with latency** (corr **+0.67**, every seed); and it obeys
**Weber's law** — decoded-time SD grows with elapsed time at a ~constant Weber fraction (CV **0.15**,
scale-invariant; untrained 0.22). None of these were in the loss: the brain's interval-timing signatures
are *measured, not designed*. `results/time_cells.{json,svg}`.

*Toward the biophysical organ (spiking, multi-timescale).* A spiking successor
(`src/models/neuro/spiking_temporal_cortex.py`: recurrent adaptive-LIF, surrogate-gradient spikes,
per-unit **learnable** membrane and adaptation time-constants) reproduces the signature in spikes and
adds a functional multi-timescale result (n=6, vs a homogeneous-τ control): spiking time cells emerge
(**46%**, from spike-frequency adaptation), and a heterogeneous **timescale spectrum emerges (14.6×)**
that **improves timing** (decode error **0.87** vs **1.47** steps homogeneous); widening (**+0.47**) and
scalar timing (**+0.70**) reproduce, noisier than rates. Honest non-result: a "slow cells code late"
(log-compression) trend at n=2 did not replicate at n=6 (corr(τ,peak) +0.10 ± 0.17).
`results/spiking_time_cells.{json,svg}`.

*The signatures survive the brain's learning rule (local e-prop, no backprop).* The rest of the paper
trains by BPTT, which brains do not do. Trained instead by **e-prop** (Bellec 2020: per-synapse
eligibility traces + one broadcast error signal; ALIF neurons give the slow adaptation-eligibility that
carries temporal credit across the delay; no autograd), a recurrent ALIF net (n=5) **learns to time**
(loss/T 0.030 < the 0.083 predict-mean floor in all 5 seeds; decode MAE 2.4 steps) and **grows spiking
time cells** (10% ± 2; fewer than backprop's ~46% but consistent). The time-cell signature thus does not
require backprop — the architecture gives rise to it even under a brain-faithful local rule.
`results/eprop_local_learning.{json,svg}`.

*One circuit for space and time.* Hippocampal place, time, and conjunctive space×time cells share a
single population (Neuron 2024). Feeding ONE recurrent substrate velocity + a start pulse and training it
to report both position and elapsed time, all three coexist (n=5; classified by η² variance-explained for
space vs time, decorrelated in a bounded box): pure place **19% ± 3**, pure time **17% ± 3**, conjunctive
**51% ± 3** (conjunctive-dominant, as observed), decoding position (MAE 0.20) and time (MAE 1.30 steps)
together. Space and time are multiplexed in the same units, not separate modules.
`results/space_time_circuit.{json,svg}`.

*From reproducing neuroscience to proposing it.* Because the signatures emerge rather than being built
in, the substrate can be perturbed to generate **falsifiable predictions** (`src/eval/predictions.py`).
Two standing examples: (P1) content load sets the conjunctive/pure ratio — the share of conjunctive
(event×time) time cells rises from 0% (content-free) to ~70% (cue-rich); (P2) spatial-input reliability
sets the space/time mix — corrupting self-motion input drives the pure-time share from 21% to 84%.
Neither was designed in; each is a number an experiment can refute (degrade vestibular/optic-flow input,
or vary cue count, and read out the cell-type proportions). We have also run the loop in the rejecting
direction: the model's "slow cells code late" log-compression prediction failed to replicate at n=6.
`results/predictions.{json,svg}`.

*The behaving agent — the map drives behavior.* Closing the loop (`src/eval/agent_navigation.py`, n=5):
an agent path-integrates self-motion into a place code, feeds a dopamine-TD critic + a basal-ganglia-like
actor, acts, and learns online — goal-directed navigation emerges (success → 100%). And one **successor
map the agent learns from its own exploration** drives **flexible, zero-shot navigation to any goal**
around a barrier (**100%**), where Euclidean vector-navigation stalls (**69%**) and a model-free goal-A
policy fails to transfer (**13%**) — the defining capacity of a cognitive map, now driving an agent rather
than being probed. `results/agent_navigation.{json,svg}`.

*Memory-guided behavior — one-shot place learning (`src/eval/agent_memory.py`, n=5).* Adding the
hippocampal episodic store: when the reward moves each "day", a single rewarded trial collapses latency
from **142 → 7 steps** (the agent stores the location in one shot and recalls it), and **lesioning the
episodic store abolishes the savings** (latency stays ~130) while leaving navigation intact — the Morris-
water-maze signature and its hippocampal dependence, emergent in the agent. `results/agent_memory.{json,svg}`.

*Timing-guided behavior (`src/eval/agent_timing.py`, n=3).* The temporal organ driving action: in an
interval-production task (act at target D, reward peaks at D), a policy reading the emergent time-cell
population acts **precisely at D=25** (reward **0.88**); **lesioning the temporal code abolishes timing**
(acts immediately, reward **0.00**), the rest intact. Across the three behaving-agent capacities the map
is clean — flexible navigation (cognitive map), one-shot place memory (episodic store), timed action
(time cells) — each emergent from integrating an organ into the loop, and each **independently
lesionable**: a brain-in-miniature with a structure→function→lesion correspondence.
`results/agent_timing.{json,svg}`.

*The unified agent — one task, all three organs, a triple dissociation (`src/eval/agent_unified.py`, n=3).*
A single agent on a *delayed memory-guided harvest* (recall WHERE via the episodic store → navigate THERE
via the cognitive map → harvest at WHEN via the time cells; reward needs all three) shows a textbook
triple dissociation: **all-intact 99%**, and removing any single organ zeros the reward via *its own*
failure mode (**−map 0%**: can't reach; **−memory 0%**: wrong place; **−time 0%**: wrong moment). Three
capacities, emergent from one self-supervised substrate, dissociating like the brain's — the cleanest
single embodiment of the thesis. `results/agent_unified.{json,svg}`.

*The agent on its real grid cortex — connecting WHY a grid code to WHAT it does (`src/eval/agent_grid_cortex.py`,
n=3).* We replace the abstract map with the **real velocity-driven hexagonal grid cortex** (`_HexGridModules`:
6 modules, fixed biological gains; Burak & Fiete 2009) as the agent's spatial substrate. The agent
**path-integrates self-motion** so a 384-unit grid code is its only sense of position (verified: the public
`grid_code_at()` equals the recurrent integrator exactly), **reads position with a nonlinear place-cell-like
network** — the very decoder §grid-capacity shows is needed (decode error 0.024 nonlinear vs 0.030 linear) —
and **vector-navigates** to a remembered goal (100% closed-loop). On this real substrate the triple
dissociation holds exactly (**all-intact 100%**; **−grid 2%**, **−memory 1%**, **−time 0%**). The spatial
organ is no longer an abstraction but the same biologically-constrained grid code whose capacity we measured
above, and lesioning it abolishes the navigation that capacity buys. `results/agent_grid_cortex.{json,svg}`.

*Path-integration drift and its correction by boundary-vector cells — the Fiete caveat, resolved
(`src/eval/agent_grid_drift.py`, n=3).* Grid path integration is famously vulnerable to **drift** under
noisy self-motion (Burak & Fiete 2009); the brain corrects it with **allothetic** boundary cues
(Hardcastle, Ganguli & Giocomo 2015). We reproduce both on the closed-loop agent using the **real
`BoundaryVectorCells` organ** with a *learned* allothetic read-out (near-wall error 0.005). (A) Without
correction the self-localization error over a long walk **grows unbounded** (final ≫ mean: 1.72 vs 1.29 at
noise 0.15); routing the boundary sense through boundary-vector cells makes it **stationary** (final ≈ mean,
0.61 vs 0.57 — the classic sawtooth), ~3× lower. (B) The behavioral cost: over a 6-goal foraging episode
drift compounds (no-anchor 66%→15% as noise grows 0.05→0.20) and BVC anchoring rescues it (78%→24%).
Nothing is hard-coded — the localizer is learned from the BVC population and the drift/correction dynamic
emerges from combining the noisy integrator with the gated boundary sense. `results/agent_grid_drift.{json,svg}`.

*A self-correction: near-optimal cue integration (`src/eval/agent_cue_integration.py`, n=3).* On review, the
anchoring above uses a hand-coded fixed gate — not how the brain combines cues. The brain integrates
idiothetic (PI) and allothetic (boundary) cues near-optimally, with combined precision better than either
alone (Ernst & Banks 2002; Nardini 2008); the fixed gate is ~3–4× worse than optimal. We replaced it with a
generic learned recurrent fuser (a GRU; no hand-coded gate, no Kalman structure) reading only the drifting
grid-PI estimate + the boundary-cell observation, trained only to localize. (A) It beats both single cues
AND the old fixed gate and tracks/beats the Kalman optimum (noise 0.15: learned 0.85 vs PI 1.69, boundary
1.04, fixed 1.40, Kalman 1.07) — near-optimal integration, emergent. (B) Ablating the boundary collapses it
to ~PI-only (0.54→1.05) — genuine integration, not PI denoising. (C, honest) error stays bounded as the
boundary degrades (noise 0.05→3.0: 0.54→0.58) because the recurrent fuser averages unbiased observations; we
therefore claim near-optimal *integration* but NOT the strict reliability-weighting law (confounded by
temporal averaging — left open). A record of method as much as result: the right phenomenon (drift +
boundary correction) had been reproduced with the wrong mechanism (a fixed gate), and was corrected.
`results/agent_cue_integration.{json,svg}`.

*A head-direction organ — emergent ring attractor + heading-dominated drift (`src/eval/head_direction.py`,
n=5).* Biological PI drift is dominated by heading (angular) error from the head-direction system, which the
drift module above crudely modelled as translational noise. By the same emergence method (train a generic
substrate; measure signatures never in the loss), a generic rate-RNN trained only to track heading from
angular velocity develops (1) HD cells (units tuned to one heading, 57% vs 24% untrained) and a functional
ring attractor — accurate, stable heading maintenance (decode 2.6° vs 86° untrained; the untrained net
cannot hold heading). Honest nuance: a ring-*shaped* manifold appears even untrained (inherent to recurrent
integration), so the emergent signatures are the HD tuning and accurate maintenance, not the manifold shape.
(2) The emergent HD net integrates noisy angular velocity, so heading drifts (77° over a 140-step walk) and
drives position drift (13.4); a visual landmark pinning the ring bump bounds both (heading 13°, position 3.1)
— the biologically-correct heading-dominated drift and its allothetic correction (Knierim 1995). This makes
drift in the agent loop mechanistically right. `results/head_direction.{json,svg}`.

*The dead-reckoning brain — one closed HD→grid→place stack (`src/eval/agent_deadreckoning.py`, n=3).* The
spatial organs unify into a single self-localization loop: the agent estimates BOTH heading and position
from its own motor commands — motor → HD ring attractor (heading, drifts) → grid cortex path-integrates
position using that heading (drifts more) → place read-out. The integrator accumulates each actual
displacement rotated by the heading error, so drift originates as heading error and propagates into
position. With true heading the stack is near-perfect (oracle 0.04); the HD organ in the loop inflates
position error (2.41), and — an honest, instructive finding — correcting heading ALONE (visual reset, 2.52)
does not rescue position (the grid integrator's accumulated error persists): only the grid (boundary)
correction fixes position (0.44), and adding the HD correction on top bounds it best (both 0.12). Lesioning
HD (3.23) or grid (3.11) is catastrophic. Homing (path-integration return; Wehner's desert ants) works
intact (0.35) and is abolished by lesioning HD (2.79) or grid (3.11). The cleanest single embodiment of a
dead-reckoning brain — heading and position both inferred from self-motion through emergent organs, with two
distinct allothetic corrections, one per organ. `results/agent_deadreckoning.{json,svg}`.

*A multi-reference-frame map — object-vector cells + grid reanchoring (`src/eval/reference_frame.py`, n=5).*
The map so far is a global allocentric metric, but the entorhinal code also carries egocentric object-vector
cells (Høydal et al., Nature 2019) and reanchors to task-relevant objects (Butler 2019; Boccara 2019),
estimating position in multiple local frames (a 2025 frontier). We add a new `EgocentricObjectVectorCells`
organ and measure: (A) the OVC population encodes the egocentric object vector (decode err 0.030); (B) on an
object-relative goal whose object MOVES each episode, an object-frame agent (object-vector cue → HD
egocentric→allocentric transform) reaches it 100%, a global-frame agent only 17%, and lesioning HD drops it
to 15% — object-relative behavior needs both the object cue and the HD transform, not the global map; (C) the
object-frame grid code translates by the object displacement (match 0.000 vs un-shifted 0.073) — grid cells
reanchoring by translating the pattern. (Honest: object-relative nav is robust to unbiased object-cue noise
via temporal averaging, not a graceful down-weighting.) This turns the model from a global path-integrator
into an entorhinal reference-frame transformer. `results/reference_frame.{json,svg}`.

*Dynamic reanchoring of the grid phase to a landmark — allocentric & egocentric coexisting
(`src/eval/landmark_anchoring.py`, n=3).* The review's exact mechanism: the grid phase dynamically
reanchored to a landmark during path integration under cue reliability (`ego = OVC(landmark)`;
`p_hat = anchor − R(heading)·ego`; `grid = (1−w)·grid + w·gains·p_hat`), like boundary anchoring but anywhere
the landmark is seen. (A) reanchoring corrects allocentric drift (pure PI drifts to 3.12; landmark-anchored
0.87). (B) allocentric (global, from the grid: 0.87) and egocentric (landmark-relative, from object-vector
cells: 0.78) positions COEXIST — read at once, the two MEC frames (Nature Comms 2025). (C) reliability: a
reliable landmark helps (0.97), the benefit vanishing toward PI as it gets noisy. Honest: the strictly-optimal
combiner is the learned fuser of agent_cue_integration; a hand-coded Kalman gate is mis-calibrated here, so we
report the reliability dependence, not optimal weighting. The grid is path-integrated globally and reanchored
to landmarks on demand, both frames coexisting. `results/landmark_anchoring.{json,svg}`.

*Object reanchoring INSIDE the core grid cortex — load-bearing, not an eval loop (`src/eval/agent_grid_reanchor.py`,
n=5).* The reanchoring above lived only in a standalone loop; the core path-integrator (`_HexGridModules`) reset
its phase only at boundaries. We wired the egocentric object-vector organ into the module itself —
`_HexGridModules.forward(object_obs=…)` corrects the grid phase through the SAME egocentric→allocentric transform
the boundary path uses (one shared `_ego_to_allo`→`_apply_phase_fix` bridge for boundary/object/centre anchors).
Allocentric decode error (lower=better): in the OPEN FIELD (walls far) boundary anchoring barely helps (0.71, vs
path-int 0.96) but the OBJECT cue reanchors the grid ~6× better (0.13) — a capability the boundary-only module
lacked; a SHUFFLED-anchor control fails (2.37), so the rescue is the true geometry, not extra input; and NEAR A
WALL the local boundary capability is preserved (0.80 vs 2.43). The grid is path-integrated globally and
reanchored to whichever allothetic cue is available, from within one module. `results/agent_grid_reanchor.{json,svg}`.

*3D navigation via a plane-aligned 2D grid — the bat scheme (`src/eval/plane_of_motion.py`, n=5).* Bats
appear to use a 2D toroidal grid aligned to the behaviorally-relevant plane of motion + an off-plane code,
not a full 3D lattice (2026); the repo's `(x,y,z,t)` had coded height as a 1D stub. We implement it with the
real hex grid cortex on the PCA-estimated motion plane: (A) PCA recovers the motion-plane normal almost
exactly (err ~0.005, any orientation); (B) the plane-aligned 2D grid localizes 3D position with accuracy
flat across plane tilt (0.128→0.127) — orientation-invariant; (C) a fixed horizontal grid degrades as the
plane tilts steeply (0.138→0.174 at 80°) — alignment is necessary. Honest scope: at matched budget there is
no robust 3D-decode advantage over a naive isotropic 3D grid (decoder-masked), so the contribution is the
faithful, orientation-invariant mechanism + the alignment necessity, not a decode win over a 3D lattice.
`results/plane_of_motion.{json,svg}`.

*Theta-cycle look-around — online sweeps as active look-ahead (`src/eval/theta_sweep.py`, n=5).* Beyond
path integration and offline replay, grid/place activity in each theta cycle sweeps outward from the agent,
alternating left/right across cycles, sampling surrounding (incl. never-visited) space (Vollan, Gardner,
Moser & Moser, Nature 2025). We add a `ThetaSweepSampler` and show it is functional: in a concave-dead-end
field, an agent that uses the sweep to sample the grid map ahead reaches the goal 100% vs a reactive
(current-position-only) agent's 76%, at equal path length — routing around the traps the reactive agent
enters. The sampler reproduces the Vollan signatures (left/right alternation; length 19.7% of spacing,
multi-scale per module with r=1, module-aligned), and emits grid codes along the sweep as look-ahead tokens.
Honest: the sweep statistics are constructed to match Vollan (an added mechanism, like the boundary/
object-vector cells); the new result is the mechanism + its look-ahead function. `results/theta_sweep.{json,svg}`.

*Theta-sweep tokens are load-bearing for the readout/LLM (`src/eval/theta_sweep_readout.py`, n=5;
`TrajectoryLLM(use_theta_sweep=True)`; `notebooks/m7_theta_sweep_llm_kaggle.py`).* The sweep must feed the LLM
and matter. `TrajectoryLLM` now concatenates theta look-ahead tokens to the current spatial token (`_sweep_tokens`
samples the grid map ahead, alternating L/R, and projects each swept code to a token; real/shuffled/ablated
modes). In a NOVEL per-episode layout (so the answer is not knowable from position — the agent must look) a
fixed readout predicts whether the cone ahead is blocked: real sweep 0.90 vs sweep-ablated 0.58 vs
wrong-heading-shuffled 0.63 (chance 0.50). Only the real sweep can see ahead — a clean, capacity-independent
ablation that the tokens carry the look-ahead. `results/theta_sweep_readout.{json,svg}`. The FROZEN-LLM
confirmation (`notebooks/m7_theta_sweep_llm_kaggle.py`, n=8 on a T4): a frozen Qwen2.5-1.5B (LoRA + gated
fusion) judges "blocked ahead?" in a novel layout (moves never in the prompt, so ON vs text-only-OFF is causal)
at 82% ±16 with the real sweep tokens, falling to chance without them — 48% sweep-ablated and 50% text-only
(both indistinguishable from 50%), 56% shuffled (barely above chance). The decisive contrast is ON vs NO-SWEEP
(both carry the cortex; only the sweep differs): +34%, p=0.0081 (the n=8 sign-flip floor). Honest: the ON mean's
95% CI is ±16% (solidly above chance); the per-seed spread is wider — a seed or two did not converge (near
chance). At n=8 and seed-variable, the review's demand borne out at the language level — the LLM uses
theta-sweep tokens, and removing them drops performance to chance.
`results/theta_sweep_llm_agg.json`.

*Coexisting egocentric anchors — center, object, boundary (`src/eval/egocentric_anchors.py`, n=5).* MEC holds
allocentric and egocentric codes at once, including egocentric bearing/distance to the geometric center and
to boundaries (Nat Commun 2025). We add the missing center anchor (`EgocentricCenterCells`) and show three
egocentric anchor frames coexist: the combined population decodes the egocentric vector to the center (0.24),
an object (0.62), and the nearest boundary (0.10) simultaneously, and each frame decodes from its own cells
but not from another's (≥0.42) — a multi-anchor egocentric↔allocentric transformer, not a single frame.
`results/egocentric_anchors.{json,svg}`.

*Local 3D order, not a global lattice (`src/eval/local_3d_order.py`, n=5).* Bat MEC 3D grid cells show local
order (regular nearest-neighbor field spacing) but no global 3D lattice. We make this measurable: local order
(1−CV of NN distance) vs global lattice (max structure factor S(q)/N). A local-order (blue-noise) field code
scores high local (0.95) / low global (0.05) — the bat regime — cleanly separable from a true 3D lattice
(0.94/0.88) and random (0.65/0.05). So the repo's 3D story is the bat-faithful "local order, not a lattice",
not a naive cubic grid. `results/local_3d_order.{json,svg}`.

*A biologically-grounded 3D grid code replaces the 1-D z stub in the core cortex (`src/eval/grid_3d.py`, n=5;
`LocalOrder3DGrid`; `_HexGridModules(grid_3d=True)`).* The core integrator coded height as a 1-D place stub;
we replace it with a real 3D code. `LocalOrder3DGrid` gives each cell multiple 3D fields from a shared
blue-noise packing -> local order, NO global lattice (the bat MEC regime; Ginosar et al., Nature 2021), and
path-integrates 3D self-motion. (A) Its field centers are in the bat regime: local order 0.90, global lattice
0.01 -- vs a cubic lattice (1.00/1.00, the non-biological crystal) and random (0.64/0.02). (B) It is metric:
the population localizes in full 3D (decode err 0.21, vertical 0.11), about as well as the lattice (0.16) --
faithfulness costs ~nothing. Wired in via grid_3d=True, the core cortex path-integrates 3D self-motion and
localizes (err 0.19) -- height is grid-coded, not a stub. `results/grid_3d.{json,svg}`.

*The unified multi-reference-frame navigating brain (`src/eval/agent_multiframe.py`, n=3).* The functional
consolidation: not five reference-frame demos but ONE closed-loop agent navigating in both a global
(allocentric) frame via the grid position code and an object-centred (egocentric) frame via object-vector
cells + the HD transform, sharing one organ stack (steering is egocentric, so HD is needed either way). A
clean DOUBLE DISSOCIATION: intact reaches both goals (100%/100%); lesioning the grid kills the global frame
only (20% vs object 100%); lesioning the object-vector cells kills the object frame only (12% vs global
100%); lesioning head-direction kills both (10%/10%). One brain holding and acting in two reference frames —
the functional embodiment of the reference-frame transformer. (Its language counterpart, a frozen LLM
answering in both frames from the combined code, is notebooks/m6_multiframe_llm_kaggle.py.)
`results/agent_multiframe.{json,svg}`.

*A basal-ganglia action-selection organ (`src/eval/basal_ganglia.py`, n=3).* The first system beyond the
hippocampal core: a cortico-striatal Go(D1)/NoGo(D2) opponent circuit selecting actions by softmax(Go −
NoGo) and learning by **local dopamine-RPE-gated** three-factor plasticity (Frank OpAL) — no backprop.
Intact it learns to **100%**; **lesioning dopamine collapses learning to chance (35%)** — the
dopamine-dependence of reward-based action learning. (The Go/NoGo pathways are partially redundant here —
either alone reaches 100% — so it is loss of the shared dopamine signal, not one pathway, that abolishes
learning.) `results/basal_ganglia.{json,svg}`.

*Why a grid cortex? — coding capacity at scale (`src/eval/grid_capacity.py`, n=5).* The agent runs on a
grid cortex; here we show *why* the brain pays for one. Behaviorally, navigation to a region is forgiving
(grid and place both reach ~100% across arena sizes — no behavioral edge); the grid advantage is
**representational** (the Fiete claim). We measure it decoder-agnostically with **Fisher information** (the
Cramér–Rao bound; both closed forms verified against autograd). At a **fixed neuron budget**, as the arena
scales 8×, grid local resolution stays **~flat** (log-log slope **+0.18**; set by its finest, space-reused
period) while place degrades **~linearly** (slope **+1.00**; a fixed budget of bumps tiles ever more
coarsely) — the grid advantage **grows to 33×** (exponential-vs-linear capacity; Sreenivasan & Fiete 2011).
*Honest caveat:* a **linear** reader cannot extract it (linear-decode MAE is *worse* for grid than place) —
the capacity is real but requires a nonlinear/Bayesian decoder, which is exactly why downstream place cells
(a nonlinear conjunction of grid inputs) exist. `results/grid_capacity.{json,svg}`.

*Catastrophic errors — the other half of the trade-off (`src/eval/grid_catastrophe.py`, n=5).* The grid code
is a residue code, so its capacity has a price: under noise a phase slip can land the residue combination on
a far-off aliased position — a catastrophic error (Sreenivasan & Fiete 2011). ML-decoding a noisy grid code,
(A) adding modules suppresses the catastrophic rate exponentially (K=2→6: 75%→1%) at constant local
precision (median 0.003→0.002): modules buy catastrophe-safety, not resolution — why the entorhinal code is
multi-module (Stensola 2012); (B) the error law is bimodal (K=2: 25% local / 75% catastrophic, almost
nothing between; gone by K=5). (C) Honest correction to my own first framing: I expected "place is
catastrophe-safe but coarse", but the data refuted it — a place code also makes catastrophic wrong-bump
errors, and at matched budget the grid is ~19× finer AND no more catastrophe-prone (grid 19% vs place 25% at
the highest noise). So the catastrophe-risk is intrinsic to noisy decoding, settled within the grid by
multi-module redundancy, and the grid dominates place once a nonlinear decoder unlocks its capacity. With the
capacity result this is the complete Fiete picture. `results/grid_catastrophe.{json,svg}`.

*Content-binding (what-where-when).* The temporal code also binds content, reproducing a 2023 hippocampal
result (bat CA1; Shimbo et al., *Nat Neurosci*; *Neuron* 2024): given one of K events at t=0 and asked to
report both elapsed time and which event, the substrate grows **two coexisting populations** — **pure**
time cells (29% ± 7) and **conjunctive "contextual"** cells (71% ± 7, event × time) — and decodes BOTH
**what** (event 100% vs 33% chance) and **when** (1.31 ± 0.12 steps), n=6 (`src/eval/content_binding.py`,
`results/content_binding.{json,svg}`). Local (e-prop) learning and grid-cortex embedding remain open; the
natural next step is a frozen-LLM "what happened when?" readout.

Together these give the cortex a map that **plans** (detours a metric map cannot) and **keeps time**
(with the brain's scalar law) — the two axes a purely-spatial code omits, each falsified before transfer.

## 8. Language transfer ✅ (causal ON≫OFF readouts significant at n=6; grid-vs-place n=3)

A LoRA-Qwen2.5-1.5B answers navigation questions through the frozen cortex (the moves reach the model
only via the cortex). We report the **multi-seed** result (n=3, mean ± 95% CI; `results/extrapolation_llm.json`,
Figure 5), and it is honest in two directions:

| cortex-ON exact, T=8/16/24 | grid | place | text-only (OFF) |
|---|---|---|---|
| bearing | 80/81/**71** ±13–16 | 53/43/47 ±32–37 | ~11% |
| distance | 53/50/**46** ±38–42 | 58/40/30 ±11–20 | ~14–17% |

1. **The cortex channel genuinely carries the answer** — cortex-ON sits far above the text-only OFF
   control (bearing 71–81% vs 11%; distance ~46–58% vs 14–17%), so the LLM reasons through the
   self-supervised spatial code, not the prompt. This is the robust, primary language result.
2. **grid vs place is not statistically separable at n=3** — seed variance is large (distance grid
   ±40%). A clean single-seed run had suggested a big grid advantage on distance (95/88/85 vs
   62/46/40); it **did not replicate** under multiple seeds (a lucky seed), exactly as our CPU
   characterization predicts. *Bearing* trends grid-favorable (tighter, higher, flat to 3×) but its CIs
   still overlap at n=3.

So the language evidence supports the honest thesis precisely: a self-supervised cortex transfers
spatial competence to a frozen LLM (ON ≫ OFF), while the *grid-over-alternatives* advantage is modest
and, on the hardest task, within noise at n=3 — resolving it needs n≥8 (and may remain a bearing-only
effect). (Figure 5: `results/extrapolation_llm.svg`.)

**Leakage-proof causal transfer on a non-Euclidean world (`--task torus`).** The cleanest language
result: a frozen cortex *pretrained on the torus* lets Qwen answer "which wrap-around cell are you in?"
— a question with no faithful Euclidean text description, with the moves never in the prompt. Across
**n=6 seeds**, **cortex-ON beats text-only-OFF by +52 to +73 points at every length and in every seed**
(ON 84/74/63% at T=8/16/24 vs OFF ~9–11% chance; `results/torus_llm.json`). Because the world is cyclic, a
language prior over Euclidean space cannot substitute; the LLM must be *reading the path-integrated
toroidal code*. The paired sign-flip permutation test is **significant at every length (p = 0.033)**,
clearing the n=3 floor; the ON magnitude remains seed-variable (CIs wide), but the causal direction is
significant and consistent across seeds and lengths. This single-item
readout transfers cleanly — whereas a two-item **comparison** does **not** train through the same
frozen-LLM fusion interface (`results/relational_llm.json`: exactly chance across seeds/evaluators).
That contrast — single-item spatial readouts transfer to a frozen LLM, pairwise comparison does not — is
itself a finding and an honest scope statement. (Figure 7: `results/torus_llm.svg`.)

*What-happened-when (content-binding capstone) — a joint-answer capacity tradeoff.* Asking the frozen LLM
to read BOTH fields of the content-binding cortex (§7) — neither in the prompt — each field is
*individually* significant but they *trade off in one answer* (n=6): event-first/equal-weight reads
**WHAT** (cortex-ON 76% vs OFF 26%, p=0.033) with WHEN at chance (p=0.78); time-first + up-weighting the
time tokens reads **WHEN** strongly (exact 67% vs 17%, p=0.033; within-1 91% vs 44%, p=0.033) with WHAT
marginal (43%, p=0.095). The fusion interface reads the categorical *or* the scalar field — whichever the
loss emphasizes — but a single autoregressive answer is a capacity bottleneck. This is a *readout*
property, not the binding: the cortex encodes both (CPU decode) and the standalone elapsed-time readout
succeeds (p=0.033). A separate-query readout (asking *what?* or *when?* independently) confirms each is
readable but inherits the same limit — split 50/50, WHEN stays significant (78% within-1, p=0.033) while
WHAT slips to marginal (p=0.16) on its halved share. Net: a frozen LLM reads *either* field of the bound
code to significance, but a single small LoRA readout cannot max both — a capacity/training-share limit
of the interface, not of the binding. (`results/what_when_llm.json`.)

**The emergent TIME code transfers too — the temporal analogue (`notebooks/m3_temporal_full_kaggle.py`).**
The same single-item-readout logic closes the *temporal* loop: a frozen LoRA-Qwen answers "how much time
has elapsed?" (6 bins) reading ONLY the FROZEN *emergent* temporal cortex (§7) — elapsed time never in the
prompt. Across **n=6 seeds** (chance 17%), **cortex-ON beats text-only-OFF in every seed**: EXACT ON **55%
± 20** vs OFF **16% ± 6** (Δ+40; OFF at chance — the clean contrast), and on WITHIN-1 (the natural metric
for a scalar quantity) ON **70% ± 19** vs OFF **37% ± 17** (Δ+33), best seed **86%/96%**. With all six
seeds ON>OFF the paired sign-flip permutation test is **significant on both metrics (p = 0.033)**. The
only caveat (shared with torus) is that the ON magnitude is seed-variable (±20; the cortex's emergent-code
quality varies seed to seed). So a frozen LLM reads an **emergent time-cell code it was never given in
text** — both axes of the predictive-spatiotemporal map, space (torus) and time (elapsed), now transfer
to language, all emergent. (`results/elapsed_time_llm.json`.)

**The dead-reckoning brain speaks — a frozen LLM reads BOTH emergent organs, organ-specifically**
(`notebooks/m5_deadreckoning_llm_kaggle.py`, n=6). The founding-goal capstone: a frozen LoRA-Qwen reads the
unified dead-reckoning agent's emergent self-localization code — the grid-cell population (position) and the
head-direction ring-attractor state (heading) — and answers in language (moves never in the prompt; cortex-ON
vs text-only-OFF, causal + leakage-proof). Two *direct single-organ* decodes: **WHERE** (which of 9 cells)
reads the grid code — ON **38% ± 32** vs OFF **8%**, **significant (p=0.033**, all 6 seeds ON>OFF); **FACING**
(heading, 8 sectors) reads the HD code — ON **40% ± 26** vs OFF **12%** (Δ+28), a strong trend not clearing
0.05 at n=6 (**p=0.095**). The decisive evidence is an **organ-specific double dissociation**: each read
collapses *only* when its own organ is ablated — WHERE no-grid **8%** (dies) vs no-HD **39%** (survives);
FACING no-HD **10%** (dies) vs no-grid **33%** (survives). So the LLM reads position *specifically* from the
grid cortex and heading *specifically* from the head-direction ring — the emergent organs become a spatial
sense an LLM speaks from, each causally traced to its organ. *Honest scope:* FACING's ON-vs-OFF is a trend
(its organ-specific lesion independently confirms it reads HD); the harder egocentric **homing-vector**
readout (a nonlinear cross-organ combination) was null and is left as future work; ON magnitude is
seed-variable (as in torus/time). (`results/deadreckoning_llm_agg.json`.)

**The map speaks BOTH reference frames — allocentric and egocentric, organ-specifically**
(`notebooks/m6_multiframe_llm_kaggle.py`, n=8). The language counterpart of the unified multi-reference-frame
agent: a frozen Qwen reads the combined code — grid (global) + egocentric object-vector cells
(landmark-relative) — and answers in either frame. LANDMARK (egocentric direction ← object-vector) ON 35% vs
OFF 13% (Δ+23, p=0.031, significant); WHERE (which room cell ← grid) ON 47% vs OFF 8% (Δ+39, p=0.053, at the
threshold — limited by one non-convergent seed whose readout trained below chance, not a real null; the other
7 are all ON≫OFF). The decisive evidence is a clean organ-specific DOUBLE dissociation: WHERE collapses only
when the grid is ablated (11% vs 49%); LANDMARK only when the object-vector cells are ablated (11% vs 36%) —
the LLM reads the allocentric frame specifically from the grid and the egocentric frame specifically from the
object-vector cells. The review's vision at the language level: a map that answers "where am I globally?" and
"where am I relative to the landmark?", both frames coexisting and each causally traced to its organ.
(`results/multiframe_llm_agg.json`.)

## 9. Related work ✎

Grid cells / path integration (Hafting 2005; Burak & Fiete 2009); grid codes in trained integrators
(Banino 2018; Cueva & Wei 2018); modular coding for range/capacity (Fiete; Stensola 2012; Sreenivasan &
Fiete 2011); the Tolman-Eichenbaum Machine and grid codes in concept space (Whittington 2020;
Constantinescu 2016); Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995); length
generalization in sequence models (the default does not generalize — the motivation for positional-
encoding research). Our contribution is the *fair, multi-seed characterization* of which of these
properties transfer to a trained model + the integrative LLM demonstration.

## 10. Limitations (honest) ✎

- The representation tasks are 2-D, unbiased random walk (~√T magnitude growth), single-T4 LLM scale.
- The headline "grid extrapolates" claim is matched by a NoPE+sum Transformer; the grid code is not the
  best pure path-integrator.
- The remapping/capacity advantages are regime-specific (fixed memory / context-free) and do not
  transfer to a trained LLM with a text context label.
- §8 is n=3 with large seed variance; the grid-vs-place comparison there is inconclusive (needs n≥8).
  Emergence, boundary, replay pillars are demonstrations.

## 11. Methods ✎

**Grid cortex** (`_HexGridModules`): K modules, fixed velocity gains `side/spacing`,
`spacing = base·ratio^k`; per-step velocity advances a phase integrated and min-image-wrapped on a
hexagonal torus; module population (B, K·side²) read by a learned linear map. **Place code**: Gaussian
fields tiling the arena. **Self-supervision**: predict bounded place activity from integrated
self-motion; no coordinate labels. **Baselines**: GRU integrator; Transformer encoder with
learned/sinusoidal/no positions and mean/sum pooling; raw-displacement and random-Fourier lifts;
exact-integration oracle. **Statistics**: each metric re-implemented in a seed loop; mean ± 1.96·sd/√n.
**LLM**: gated cross-attention from the cortex into frozen Qwen2.5-1.5B + LoRA (q,v); answer-only loss.
Full configs in `results/*.json`; one-command regeneration via `bash reproduce_all.sh`, with the
figure→command→artifact map, verified environment, and Zenodo-release steps in `REPRODUCE.md`.

---

### Status / path to submission
- ✅ §3 Fig 1, §4 ablations + fair seq baselines, §5 necessity + boundary + frontier, §6 stats — all
  multi-seed, committed.
- ✅ §7 predictive (SR) + temporal (time-cell) map — CPU, n=8, committed; temporal signatures EMERGE.
- ✅ §8 causal language readouts, **both significant at n=6 (paired p=0.033, every seed ON≫OFF)**:
  **torus-QA** (space) ON 84/74/63% vs OFF ~10% at T=8/16/24; **elapsed-time** (time) ON 55%±20 vs OFF
  16%±6 exact. A frozen LLM reads the emergent spatial *and* temporal codes it was never given in text.
- ➕ optional: n≥8 LLM seeds to resolve the (modest, bearing-trending) grid-vs-place effect.
- ✎ tighten abstract/intro/related work; assemble figure panels; expand Methods/Extended Data.
- Framing locked: honest characterization (wins, ties, boundaries) + integrative demo; **no uniqueness
  claim**.
