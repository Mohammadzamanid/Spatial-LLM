# Supplementary Results — Spatial-LLM

This file preserves the extended neuroscience-inspired analyses and auxiliary demonstrations that were removed from the flagship main-text spine for focus. These results remain part of the project record and may be cited from Supplementary Information, but they are not required for the central causal cognitive-map claim.

The main manuscript prioritizes four questions: (i) causal spatial-state transfer to language, (ii) when structured spatial coding is mechanistically useful versus replaceable by simpler integration, (iii) cross-backbone replication of the strongest prospective effect, and (iv) external validation on real human trajectories.

---

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

*One-shot learning the biological way — BTSP and its predictive place field (`src/eval/btsp.py`, n=5).* The
model's one-shot memory writes a place code into an episodic store (an abstraction); the hippocampus instead
imprints a complete place field in ONE traversal from a single dendritic plateau, via a seconds-wide,
temporally ASYMMETRIC plasticity kernel (behavioral-timescale synaptic plasticity, BTSP; Bittner, Milstein &
Magee, Science 2017). We add a `BTSPPlasticity` organ, fire one plateau at the track centre, apply it once, and
MEASURE the field. (A) one-shot field formation needs a SECONDS-scale kernel: BTSP and a symmetric-seconds
control imprint a strong field in one pass (strength 1.00, 0.98) while a millisecond STDP-scale kernel imprints
almost nothing (0.02). (B) the PREDICTIVE shift needs the ASYMMETRY: only BTSP shifts the field upstream of the
plateau (−13, the cell fires before the induction site) while the symmetric control sits on it (+0.1) — the
shift is not put in, it emerges from potentiating the upstream inputs the animal traversed in the seconds
before the plateau. (C) the shift scales with running speed (−8 → −17 as v = 15 → 40), a temporal kernel read
as a spatial shift — a specific Bittner prediction. The biological one-shot rule, with its signature, emergent.
`results/btsp.{json,svg}`.

*One circuit for space and time.* Hippocampal place, time, and conjunctive space×time cells share a
single population (Neuron 2024). Feeding ONE recurrent substrate velocity + a start pulse and training it
to report both position and elapsed time, all three coexist (n=5; classified by η² variance-explained for
space vs time, decorrelated in a bounded box): pure place **19% ± 3**, pure time **17% ± 3**, conjunctive
**51% ± 3** (conjunctive-dominant, as observed), decoding position (MAE 0.20) and time (MAE 1.30 steps)
together. Space and time are multiplexed in the same units, not separate modules.
`results/space_time_circuit.{json,svg}`.

*A self map and an other-agent map in one population — social place cells (`src/eval/social_space.py`, n=5).*
The hippocampus encodes not only the animal's own position but another individual's, in dedicated social place
cells (Danjo 2018; Omer, Las & Ulanovsky 2018 in bats), and humans map social variables with the same machinery
(Tavares 2015; Park 2021) — a representation the model lacked entirely. Feeding ONE recurrent substrate its own
self-motion AND its observation of another agent's motion, and training it to report both positions, separate
populations emerge (η² by self- vs other-position, nothing imposed): pure SELF-place 22% ± 4, pure OTHER-place
20% ± 2, conjunctive 42% ± 6. They dissociate cleanly: lesioning the other-place cells wrecks decoding of the
other agent (MAE 0.21 → 0.40) while self-decoding survives (0.22), and lesioning the self-place cells does the
reverse — a self-map and an other-map coexisting in one circuit, the emergent social place cells.
`results/social_space.{json,svg}`.

*Goal & reward coding — a goal-vector code and anticipatory reward fields (`src/eval/goal_vector.py`,
`src/eval/reward_map.py`, n=5; designed with a research+red-team panel).* (A) A generic policy trained ONLY to
reach randomized goals from the grid code (the goal enters only as grid_code_at(goal), never a decoded goal
vector) navigates at 99.7% and 95% of its hidden units then tune to the direction to the goal — emergent and
goal-specific (untrained baseline 2%, goal-shuffle null 1%; the Banino-2018 vector-to-goal template). Honest
scope: the code is allocentric and redundant, and egocentric/metric-distance cells do not emerge from a
magnitude-free directional task (a noted extension). (B) Reward-triggered BTSP builds place fields that
ANTICIPATE the goal: they sit upstream of the reward along the approach (−0.23 ± 0.03) — emergent, since the
plateau fires AT the reward and only the kernel asymmetry shifts the fields before it — and this cleanly
vanishes under a symmetric-kernel control (+0.02 ± 0.03); the fields also concentrate at the reward 43× vs a
yoked random-plateau control (0.8×). The predictive reward map of Hollup 2001 / Gauthier-Tank 2018, from BTSP.
`results/goal_vector.{json,svg}`, `results/reward_map.{json,svg}`.

*A grid code for concepts — the hexadirectional signal, symmetry inherited from the lattice
(`src/eval/hexadirectional.py`, n=5).* Humans show a six-fold entorhinal signal moving through space and through
abstract 2-D concept spaces (Doeller 2010; Constantinescu, O'Keefe & Behrens 2016). Done non-circularly: a
summed grid rate map is direction-invariant, so the 6-fold lives only in the direction signal, through a
movement-sensitive nonlinearity (conjunctive grid×direction cells with UNIFORM preferred directions — nothing
6-fold imposed). Measuring the population's movement-driven activity power vs run direction, the model's
hexagonal grid gives a 6-fold signal (A6 0.040, index 80%) above the 4-fold (0.010) and the adjacent 5/7-fold
control (0.011); its symmetry is INHERITED from the lattice — a square lattice flips it to 4-fold (index 10%);
and a linear read-out is direction-invariant (A6 0.005). Reading the two axes as concept features, the same grid
metric produces the hexadirectional signature for movement through concept space — the cognitive map from space
to meaning. `results/hexadirectional.{json,svg}`.

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
at 68% ±14 with the real sweep tokens, dropping to chance without them — 41% sweep-ablated, 44% text-only,
51% wrong-heading-shuffled (all within CI of 50%). The decisive contrast is ON vs NO-SWEEP (both carry the
cortex; only the sweep differs): +27%, and all three ablations are at p=0.0081 — the n=8 sign-flip floor — so ON
exceeds every ablation in every one of the 8 seeds (unanimous). Honest: the accuracy is modest (68%; this
few-token frozen-LLM reader is weaker than the CPU readout's 0.90), and this convergence-hardened config (2800
steps) is more consistent but lower than a shorter pilot (82% ±16, ON-vs-OFF p=0.030) — hardening bought
cross-seed robustness, not a higher headline. Either way the review's demand is borne out at the language level:
the LLM uses theta-sweep tokens, and removing them drops performance to chance, in every seed.
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

**Second-backbone replication — SmolLM2-1.7B-Instruct.** To test whether the theta-sweep effect was specific to Qwen, we repeated the blocked-ahead experiment on a second open-weight model family with the same cortex and the same ON / OFF / NO-SWEEP / wrong-heading controls, using forced-choice logits to avoid output-format confounds. Across **n=8 seeds**, cortex-ON reached **69.9% ±12.6** versus **49.8% ±0.3** for NO-SWEEP (**Δ=+20.1 points; 7 wins, 0 losses, 1 tie; exact paired sign-flip p=0.0156**). ON also exceeded text-only OFF (**+19.9 points, p=0.0156**) and wrong-heading sweep (**+14.4 points, p=0.0156**). One seed remained at chance in all conditions and one showed only a small ON gain, so the replication supports the **causal direction and specificity**, not uniform convergence or identical absolute accuracy across backbones. This closes the backbone-specificity concern while preserving the observed seed variability. (`results/sweep_llm_smollm2_v2.json`.)

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


