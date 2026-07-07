# Spatial-LLM — Neuroscience Gap Register

*What the most recent neuroscience of how the human brain **navigates** and **learns** says, versus what this
model currently does — a living, code-grounded list of the honest gaps and how to close each one.*

**Method.** Every candidate below was checked against the actual codebase (`src/models/neuro/*.py`,
`src/eval/*.py`), so nothing already implemented is listed as absent. Status is one of **absent** /
**partial** / **implemented-elsewhere**, with the code evidence that justifies it. Each gap comes with an
*emergence-style* experiment — a brain signature we would **measure**, never hard-code into the loss. Ordered
by (importance to "how the human brain works/learns") × (feasibility; CPU-first).

Last updated: July 2026. (Companion to `results/FINDINGS.md`, which records what is already built.)

---

## Tier 1 — shovel-ready (CPU), high value

### 1. Behavioral-timescale synaptic plasticity (BTSP) — one-shot place fields the *biological* way ✅ CLOSED (Jul 2026)
- **Status: implemented.** `BTSPPlasticity` organ + `src/eval/btsp.py` (n=5): one plateau imprints a one-shot
  field; the field shifts UPSTREAM (predictive, −13) only with the asymmetric seconds-wide kernel; the shift
  scales with speed (−8→−17); a millisecond STDP kernel imprints ~nothing (0.02). All measured, not trained.
  See `results/FINDINGS.md` ("One-shot learning the biological way — BTSP"). *Original entry below.*
- **Neuro basis.** A single dendritic **plateau potential** creates a complete place field in **one trial**,
  via a *seconds-wide, temporally asymmetric* plasticity kernel — and the new field's peak is **shifted
  backward** (predictively) from the plateau location (Bittner, Milstein, Magee 2017; Grienberger & Magee
  2022; Priestley/Losonczy 2022). This is now thought to be the hippocampus's dominant rapid-learning rule.
- **Model status: absent.** One-shot memory (`src/eval/agent_memory.py`) is done with an **episodic store** —
  a population-vector bump is written and recalled. That is a functional abstraction, *not* the mechanism:
  no plateau gate, no seconds-wide eligibility kernel, no predictive field shift. The plasticity organs
  present are STDP / Hebbian / short-term / e-prop (`synaptic_plasticity.py`) — all millisecond-scale.
- **Why it matters.** It is *the* modern answer to "how does the brain learn a place in one shot?" — and it
  predicts a specific, measurable signature the episodic store cannot produce.
- **Proposed experiment (emergence).** Add a `BTSPPlasticity` organ (plateau gate × a ~few-second asymmetric
  eligibility trace on the grid→place synapses). Induce a field with ONE plateau on ONE traversal, then
  **measure** (not train): (a) one-shot field formation; (b) the **backward/predictive shift** of the field
  peak; (c) the asymmetric kernel width (~seconds). Contrast with the STDP rule (no one-shot field) and the
  current episodic store (no shift). CPU.

### 2. A hexadirectional grid code over a **2-D conceptual** space (human abstract cognitive map) ✅ CLOSED (Jul 2026)
- **Status: implemented.** `ConjunctiveGridDirectionCells` organ + `src/eval/hexadirectional.py` (n=5). The
  model's hexagonal grid produces a **6-fold** direction signal (A6 0.040, index 80%, above the 4-fold *and* the
  adjacent 5/7-fold control) via a movement-sensitive nonlinearity — the human hexadirectional signature. Built
  NON-circularly per the scope note below: preferred directions are **uniform** (nothing 6-fold imposed), the
  symmetry is **inherited from the lattice** (a square lattice flips it to 4-fold), and a **linear** read-out is
  direction-invariant. The two axes read as concept features → the grid code for concepts (Constantinescu 2016).
  See `results/FINDINGS.md`. *Original entry + scope note below.*
- **Neuro basis.** Humans show a **six-fold (hexadirectional)** entorhinal fMRI signal as they move through
  *abstract 2-D feature spaces* — a grid code for **concepts**, not just places (Constantinescu, O'Keefe &
  Behrens 2016; Bao 2019; Park, Miller, Boorman 2021; Bongioanni 2021; Viganò 2023). The cognitive map is
  the brain's general engine for structured knowledge.
- **Model status: partial.** `src/eval/relational.py` + `structural_transfer.py` push an abstract **1-D**
  ordered structure (ranks 0..N-1) along a single concept axis through the frozen grid cortex and get
  transitive inference — good, and it cites Constantinescu 2016. But it is **1-D order**, not a **2-D
  continuous conceptual space**, and it never measures the defining **6-fold hexadirectional** signature.
- **Why it matters.** This is the most distinctively **human**, most "Spatial-LLM" finding on the board:
  the same grid metric that localizes in space should organize *meaning*. It reuses the existing grid cortex.
- **Scope note (found on audit).** Bigger than it first looks, and easy to do *circularly*. `emergence.py`
  already trains a generic net on path integration and measures emergent grid cells incl. **4-fold-square vs
  6-fold-hex** symmetry + gridness — but that is the *spatial* signature. The **hexadirectional** signal humans
  show in fMRI is a *movement-direction* modulation that, mechanistically, needs **conjunctive grid×direction**
  coding + a nonlinearity (Doeller 2010; Bush 2015) — otherwise a summed grid rate map is direction-invariant.
  So the faithful, non-circular build is: conjunctive grid×direction cells over a 2-D *concept* space, then
  measure an emergent 6-fold *direction* signal (vs 4-fold for a square-topology control) — not just relabel a
  hex grid's axes. Reclassified **CPU-hard** (was CPU-easy).
- **Proposed experiment (emergence).** Define a 2-D conceptual space (two continuous features), drive the
  frozen velocity grid cortex along "trajectories" in it, and **measure the 6-fold rotational symmetry** of
  the population code's activity vs. movement direction (the hexadirectional signature) — emergent, never in
  the loss. Falsifier: a shuffled/curved-metric control should destroy the 6-fold signal. CPU. GPU follow-up:
  the LLM answers concept-space "which is closer?" from the grid-of-concepts code.

### 3. Vectorial **goal / reward** cells (direction-and-distance to a remembered goal) ✅ CLOSED (Jul 2026)
- **Status: implemented (two parts).** (A) `src/eval/goal_vector.py` (n=5): a policy trained ONLY to reach
  randomized goals develops a goal-DIRECTION code (95% of units; emergent + goal-specific vs untrained 2% /
  shuffle 1% nulls; Banino-2018 template). Honest scope: allocentric/redundant; egocentric + metric-distance
  cells do NOT emerge from a magnitude-free directional task. (B) `src/eval/reward_map.py` (n=5): reward-
  triggered BTSP builds place fields that ANTICIPATE the goal (upstream shift −0.23, vanishing under a
  symmetric-kernel control +0.02) + reward-specific over-representation (43× vs 0.8× yoked-random). Designed
  with a research+red-team panel to defeat circularity. See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** Single neurons encode a **vector to a goal** (egocentric/allocentric direction + distance)
  — goal-vector cells in bats (Sarel, Finkelstein, Las, Ulanovsky 2017) and reward/goal coding + reward-biased
  place-field over-representation in rodents and humans (Gauthier & Tank 2018; Boccara 2019).
- **Model status: partial.** Homing to the **origin** exists (`agent_deadreckoning.py`, desert-ant homing),
  and object-vector cells point to an object — but there is no population explicitly tuned to
  **direction+distance to an arbitrary rewarded goal**, and no reward-driven field over-representation.
- **Proposed experiment (emergence).** Add goal-vector cells built from the existing grid/HD code (goal
  position − current position → egocentric vector). Train only to reach goals; **measure** the emergent
  vector tuning and whether place/grid resources **over-represent** the goal (a signature never trained). CPU.

### 4. Coding of **other agents** / social space ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/social_space.py` (n=5): ONE recurrent substrate fed self-motion AND the
  other agent's motion, trained to report both positions, develops SEPARATE **pure self-place (~19%)** and
  **pure other-place (~17%)** cells (plus conjunctive) — emergent (η² classification, nothing imposed) — with a
  clean **double dissociation** (lesion other-cells → other-decode fails, self survives; and vice versa). The
  social place cells of Danjo 2018 / Omer 2018. See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** The hippocampus encodes **another individual's** position with dedicated "social place
  cells" (Danjo 2018; Omer, Las, Ulanovsky 2018 in bats), and humans map **social hierarchies** with the same
  grid/hippocampal machinery (Tavares 2015; Park, Miller 2021).

---

## Tier 2 — CPU, integration / harder

### 5. Neuromodulatory control of **encoding vs. retrieval** and **surprise-driven reset** — made faithful & emergent ✅ CLOSED (Jul 2026)
- **Status: implemented.** `AcetylcholineGate` + `LocusCoeruleusReset` organs + `HopfieldAssociativeMemory`
  (CA3 auto-associator; Marr 1971 / Hopfield 1982 / Treves-Rolls 1994) + `src/eval/neuromodulation.py` (n=5).
  ACh sets a tonic encode/retrieve mode by SUPPRESSING recurrent recall while ENHANCING plasticity (Hasselmo
  2006): high-ACh encoding blocks intrusion of an overlapping stored memory — **overlap-specific** (excess over a
  far-pattern floor **+0.78**), and it is **recurrent contamination, not non-storage** (intrusion grows with the
  encode recurrent gain, **+0.38**, at MATCHED write energy ‖ΔW‖); the same recurrent weights **complete** a
  degraded cue in retrieval (**+0.29**, gone with W_rec off). NE surprise (Yu & Dayan 2005; Bouret & Sara 2005)
  = **novelty not change** (AUC **1.00**; a big EXPECTED jump stays at the familiar floor) and a surprise remap
  is **adaptive two-sided** vs a matched no-reset+re-encode control — learns the new env (**+0.29**) AND protects
  the old map (**+0.27**). Designed against a circularity red-team (headline = differences vs matched controls,
  not the by-construction knob). Also wired into `BrainSpatialCortex(ach=…)`. See `results/FINDINGS.md`.
  *Original entry below.*
- **Neuro basis.** **Acetylcholine** sets the hippocampus into an *encoding* vs. *retrieval* mode (high ACh →
  encode new, suppress recall; Hasselmo 2006); **noradrenaline / locus coeruleus** signals surprise/uncertainty
  and drives network reset & remapping (Yu & Dayan 2005; recent LC work 2022-2024).
- **Model status: partial.** `src/models/neuromodulation.py` has DA-style (`PredictionErrorGate`) and NE-style
  (`AdaptiveGain`, with an uncertainty estimator) modules — but they are **generic ML blocks wired only into
  `diagnose.py` / `accuracy.py`**, not the spatial cortex, and there is **no ACh encode/retrieve switch** and
  no surprise-triggered remapping as an emergent result.
- **Proposed experiment (emergence).** Gate the hippocampal readout with an ACh-like encode/retrieve signal
  and trigger reset/remap on NE-like surprise; **measure** that encoding-mode blocks intrusion of old memories
  and that a surprising cue triggers remapping — signatures, not objectives. CPU.

### 6. **Replay** used for planning & consolidation — not just present as a ripple signature
- **Neuro basis.** Hippocampal **replay** (forward for planning, reverse for credit assignment) supports
  model-based decisions and offline consolidation (Ólafsdóttir 2018; Mattar & Daw 2018 prioritized replay;
  Liu 2019 human replay).
- **Model status: partial.** A `SharpWaveRipple` organ exists, `pillars.py` shows offline experience-replay
  *consolidating a map*, and `theta_sweep` does *online* look-ahead. But there is no **reverse-replay credit
  assignment** or **prioritized forward replay for planning** as a core, measured result.
- **Proposed experiment (emergence).** Prioritized replay of stored trajectories through the SR learner;
  **measure** faster value propagation vs. no-replay and a reverse-replay credit-assignment signature after a
  new reward. CPU (extends `successor.py` + `pillars.py`).

### 7. Explicit **uncertainty / confidence** that drives behavior
- **Neuro basis.** The brain represents **posterior uncertainty** (probabilistic population codes, neural
  sampling) and confidence signals gate exploration and cue-weighting (Ma 2006; Pouget 2013).
- **Model status: partial.** Cue integration is *near-optimal* (`agent_cue_integration.py`) and grid capacity
  is quantified with Fisher information — uncertainty is thus **implicit**. `AdaptiveGain` estimates an
  uncertainty scalar but is not behaviorally coupled. No explicit "I am lost → switch strategy" confidence
  read-out.
- **Proposed experiment (emergence).** Decode a calibrated posterior width from the population; **measure**
  that behavior (explore vs. exploit, cue re-weighting) tracks it, and that it rises with path-integration
  drift and falls at a landmark reset. CPU.

---

## Tier 3 — GPU / language

### 8. The LLM reads the **conceptual-grid** map (abstract reasoning through the cognitive map)
- After gap #2, a frozen-LLM readout answers abstract "which concept is closer / between?" from the
  grid-of-concepts code — cortex-ON vs text-only-OFF — extending the cognitive-map claim from space to
  meaning at the language level. (Notebook, T4.)

### 9. LLM reasoning over **social / other-agent** space (after gap #4).

---

## Tier 4 — research-open (flagged honestly, not yet shovel-ready)

- **Representational drift & lifelong stability.** Place codes **drift over days** while behavior is stable
  (Ziv 2013; Rule 2019). The model has a `continual.py` remapping eval but does not model gradual drift or a
  read-out invariant to it. (Open: what stays stable while the code drifts?)
- **Generative world-model / imagination.** Beyond one-step look-ahead and SR: multi-step *imagined* rollouts
  and generative replay for planning-as-inference (recent successor-features / world-model framings). Partial
  via SR + theta-sweep; full imagination is open.
- **Developmental emergence.** How grid/place/HD codes **develop and stabilize** (Wills/Langston 2010). Out of
  current scope.

---

## Tier 5 — the learning substrate (how the cortex *learns*, not just what it represents)

*The register so far is strong on representation-forming rules bolted onto a backprop-trained core (BTSP, e-prop,
STDP, Hebbian/Oja, three-factor gating, ripple consolidation). The deepest remaining gaps between this and human
learning cluster in four places the earlier tiers do not reach: the credit-assignment **substrate** itself, the
**timescale structure** of the synapse, the brain's ability to **tune its own learning**, and its **non-neuronal**
learning partners. Every item is a measured-emergence experiment in the house style — never hard-coded into a loss.*

### A1. Deep credit assignment WITHOUT backprop ✅ CLOSED (Jul 2026)
- **Status: implemented (flagship).** `src/eval/credit_assignment.py` (n=5): one deep cortex module (a
  coordinate→place-code map, 2→H→H→place) trained THREE ways from a matched init — backprop (weight transport),
  **feedback alignment** (a FIXED RANDOM backward pathway; no Wᵀ, no forward/backward symmetry — the biological
  rule), and a **shuffled-feedback** falsifier (feedback re-randomised every step). Measured, not trained:
  (A) PARITY — feedback alignment reaches backprop's spatial decode (**0.106 vs 0.105**, both ≪ position-blind
  floor 0.267) and extrapolation; (B) the forward weights **align** to the fixed feedback (weight-align **+0.07**
  grown from ~0; grad-align **+0.10** vs the true gradient) — the FEEDBACK PATHWAY carries the error, modest but
  consistently positive vs the shuffled null (**~0.00**); (C) the FALSIFIER — shuffling that pathway cripples
  learning (decode **0.147 vs 0.106**, gap +0.042 ± 0.015); (D) feedback alignment learns backprop's internal
  representation (CKA **0.98**). Hand-coded forward/backward (no autograd), like `eprop_local_learning.py`.
- **Neuro basis.** Backprop's biological objections are concrete: forward/backward **weight symmetry** (a "weight
  transport" the brain has no mechanism for), a **global** error piped through every layer, and a distinct
  **backward phase**. Live substitutes: **burst-dependent plasticity** (a neuron's burst-vs-single-spike rate
  carries a local error-like signal down the hierarchy; Payeur 2021), **dendritic microcircuits** delivering error
  to apical dendrites (Sacramento 2018; Guerguiev 2017), and **prospective-configuration / predictive-coding**
  nets. Feedback alignment (Lillicrap 2016; Nøkland 2016) is the tractable instance that removes weight transport.
- **Why it mattered.** *The* deepest "how the cortex learns" gap. `DendriticNeuron` and `predictive_coding.py`
  were already present but used as **encoders**, not as the learning substrate. This makes the credit signal
  itself non-backprop and shows the spatial signatures survive it.
- **Honest scope / next.** This closes the weight-transport objection on a feedforward module; the burst-dependent
  (Payeur) and dendritic (Sacramento) realisations, and running feedback alignment inside the recurrent
  path-integration net of `emergence.py` (to show emergent *grid* cells under a non-backprop rule), are the
  follow-ups.

### B2. The multi-timescale (metaplastic) synapse ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/complex_synapse.py` (n=5). A synapse built as a Benna–Fusi CHAIN of coupled
  variables at geometric timescales, on Benna & Fusi's own random-memory benchmark. Measured, never fit:
  (A) the complex synapse forgets as a **POWER LAW** — log-log R² **0.99** ≫ semilog R² 0.73, slope **−0.47 ±
  0.01** (≈ the −0.5 / 1/√t law) — whereas a leaky **SCALAR** synapse forgets **EXPONENTIALLY** (semilog R²
  **0.99** ≫ log-log 0.81); (B) at MATCHED initial SNR the complex synapse's memory lifetime (age at SNR=1) is
  **3.3×** longer (278 vs 84); (C) dose-response — lifetime grows geometrically with chain depth (**55 → 198 →
  278** for N=3 → 5 → 7). One weight both fast-learning and long-remembering — graceful forgetting from the
  synapse itself. Distinct from #B4 (a glial gate on the learning rule); B2 is the intrinsic multi-timescale
  synapse. See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** The modern synapse-level answer to the stability–plasticity dilemma: a single synapse is a
  chain of coupled variables at many timescales (Benna–Fusi complex/cascade synapse), giving **power-law** (not
  exponential) forgetting and letting one weight be both fast-learning and stable.
- **Model status: absent.** No metaplasticity, no hidden per-synapse state (weights are scalars in
  `place_cell_memory.py` / the Hopfield store). The "no catastrophic forgetting" result rests on the CLS
  *architecture* (fast hippocampal store + slow neocortex + ripple replay), not the synapse.
- **Proposed experiment (emergence).** Replace scalar weights in the Hopfield store / place memory with 2–3-var
  cascade synapses; train a stream of overlapping maps and **measure** the forgetting curve — predict power-law
  retention vs. exponential for a scalar-weight control at matched capacity. Extends `continual.py` +
  `grid_catastrophe.py`. CPU.

### B3. Meta-learning — the brain tunes its own learning rate from inferred volatility ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/meta_learning.py` (n=5): a GRU meta-trained ONLY to predict the next
  observation, across episodes whose hazard/noise are drawn per-episode and **never given as input**, develops in
  its **frozen recurrent dynamics** (Wang 2018 meta-RL) a learning rate that adapts online. Post-hoc-fit revealed
  α (delta-rule slope): **STABLE 0.49, VOLATILE 0.59, STOCHASTIC 0.34**. (A) tracks volatility — α_volatile −
  α_stable = **+0.10 ± 0.03**; (B) the **dissociation** (the non-circular signature a "learn faster on big errors"
  account cannot make) — α *drops* under pure stochasticity (**+0.25 ± 0.05** below volatile) even though that
  block has the **highest** observation variance; (C) an untrained net is flat (**+0.00**); (D) it beats the best
  single fixed α (error ratio **0.93**). Emergent, measured, never in the loss. See `results/FINDINGS.md`.
  *Original entry below.*
- **Neuro basis.** Humans and animals raise the learning rate in **volatile** blocks and lower it in **stable**
  ones (Behrens 2007), a prefrontal **meta-RL** process; and they dissociate **volatility** (raise LR) from
  **stochasticity** (lower LR) though both inflate observation variance.
- **Model status: absent.** Neuromodulators (DA/ACh/NE) exist but none sets a learning rate from inferred
  volatility; grep finds no volatility/adaptive-LR machinery.
- **Proposed experiment (emergence).** A small recurrent controller over the grid/SR substrate on alternating
  stable/volatile blocks; **measure** whether a post-hoc-fit effective learning rate tracks block volatility
  *without being told the block* — the meta-RL signature. Reuses `successor.py` + `neuromodulation.py`. CPU.
  *(Most distinctively-human learning gap on the board.)*

### B4. Astrocyte-gated slow plasticity ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/astrocyte_plasticity.py` (n=8): a small recurrent net learns a CONTINUAL
  stream of cue→target tasks by e-prop (eligibility + broadcast); a **slow per-synapse astrocyte** trace
  `a ← ρ·a + |Δw|` gates the update `Δw ← Δw/(1+β·a)`, throttling importance-tagged synapses. Retention error on
  old tasks after the stream: ungated **0.53** → slow-astrocyte **0.44**. Decisively, it beats a **UNIFORM**
  plasticity reduction of the **matched total ‖Δw‖** (0.47) — targeting-gain **+0.036 ± 0.024** — so the gain is
  from *where* plasticity is throttled, not from throttling less; and it needs the **SLOW timescale** — a fast
  astrocyte matches its uniform control (falsifier **+0.000 ± 0.003**). The advantage over full plasticity
  **+0.091 ± 0.034** grows with task load. Honest trade-off: protecting old costs a little new-task acquisition
  (recency +0.056). Computationally kin to EWC / synaptic intelligence; the biological content is the SLOW GLIAL
  importance gate (Williamson 2024). See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** Hippocampal "learning-associated astrocytes" orchestrate encoding/retrieval (Williamson et al.,
  *Nature* 2024); the tripartite synapse regulates efficacy over **slow (seconds)** timescales; LTP depends on
  astrocytic **D-serine**, and activating astrocytes (not neurons) enhances contextual memory. Formalised as
  Astrocyte-Gated Multi-Timescale Plasticity = eligibility + broadcast (which the repo has) + a **slow glial gate**
  (which it lacks).
- **Proposed experiment (emergence).** Add a slow astrocyte gate on the e-prop eligibility trace; **measure**
  improved retention on a continual stream vs. the ungated e-prop control at matched plasticity. A ~two-line
  extension of `eprop_local_learning.py`. CPU.

### C5. Schema-accelerated neocortical learning (beyond schema *transfer*) — *partial*
- **Neuro basis.** With a compatible neocortical **schema**, new consistent facts are assimilated in **one or two
  trials, cortically** (Tse et al. 2007) — violating the strict "neocortex is always slow" CLS assumption.
- **Model status: partial.** `relational.py` / `structural_transfer.py` test whether a frozen metric *generalises*
  (transfer), not learning **speed**.
- **Proposed experiment.** Pre-train a structural schema, then **measure** trials-to-criterion for
  schema-consistent vs. schema-inconsistent new items — predict fast/near-one-shot only for the consistent case. CPU.

### C6. Representational drift with a conserved-geometry read-out ✅ CLOSED (Jul 2026)
- **Status: implemented (and hardened against a red-team that killed a first, circular version).** `src/eval/
  representational_drift.py` (n=5). The naive version (a fixed decoder fails, a re-fit reader survives) was
  **rejected as circular** — RSA over a Gaussian tiling is blind to remapping (a full remap gives *higher* RSA),
  and the "geometry reader" was within-day recalibration. The corrected, non-circular test: at **MATCHED
  single-cell drift** (cell-corr +0.15 vs +0.13), compare geometry-PRESERVING drift (field relocation) to
  geometry-DESTROYING drift (independent high-D noise), read out by a **LABEL-FREE** manifold decoder (Fiedler /
  kNN-Laplacian 1-D coordinate — no current labels). (A) the geometry read-out recovers position under preserving
  drift (**0.001**) but fails under destroying drift (**0.30** ≈ chance), gap **+0.30 ± 0.02**; (A′) a **held-out
  supervised** decoder confirms it (**0.02 vs 0.44** — even with labels, position does not generalise once the
  geometry is gone; an all-position fit would overfit and hide this); (B) a FIXED decoder degrades under any
  drift (**0.28**) while the geometry read-out survives; (C) the read-out survives even a **FULL remap** (0% cells
  conserved, **0.002**) — it reads the environment's GEOMETRY, not cell identity. Since single-cell drift is
  matched, the difference is the drift STRUCTURE (whether geometry is conserved), not how much cells changed
  (Morales 2025). See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** Single-cell tuning **drifts** while the **population geometry is preserved**, and a
  geometry-based reader corrects for drift; a fast process drives activity onto a low-D manifold that shrinks the
  drift's dimensionality (Morales et al., *PNAS* 2025); excitability is a key stability factor.
- **Proposed experiment.** Inject slow multiplicative weight fluctuations into a trained place code; **measure**
  that a fixed decoder degrades to chance while a geometry-based (relational) read-out survives — sharpening the
  Tier-4 open item into a claim. CPU.

### C7. The sleep triple-coupling ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/sleep_consolidation.py` (n=5). Replay nested in the SO→spindle windows,
  with M memories half TAGGED (strong trace) / half untagged. Measured, at matched count: (A) **SELECTIVITY** —
  the coupled regime sends **99%** of consolidation to the TAGGED memories vs a **77% proportional floor**
  (gap **+0.21 ± 0.04**); this is EMERGENT — winner-take-all competition for the scarce spindle windows selects
  the strong traces, nothing is told to prefer tags (both regimes draw reactivations ∝ trace strength). (B)
  **COORDINATION** — at matched replay count coupled replay consolidates **every** event vs uncoupled's **50%**
  (the rest wasted in cortical DOWN states). (C) **FALSIFIER** — remove the SO structure and selectivity falls to
  the proportional floor (**0.76**, gap +0.23 ± 0.01): the *nesting*, not replay, is what selects and times.
  Guarded against the by-construction trap with matched count + the emergent-selectivity framing + the no-SO
  falsifier (Latchoumane 2017; Diekelmann & Born 2010). See `results/FINDINGS.md`. *Original entry below.*
- **Neuro basis.** NREM nests slow-oscillation → spindle → ripple; that phase-locking **selects and times** what
  consolidates. The repo has a `SharpWaveRipple` organ and replay-that-consolidates, but not the oscillatory
  nesting that gates it.
- **Proposed experiment.** Drive replay only within simulated spindle-troughs nested in slow oscillations;
  **measure** stronger, more selective consolidation than ungated replay at matched replay count. CPU.

### D8. Cerebellar supervised learning / internal forward models — *absent* (whole system; lowest priority here)
- **Neuro basis.** A third learning system: supervised, error-corrective, climbing-fiber teaching signals learning
  forward models for prediction/timing. The biggest single-system omission, but least aligned with the Spatial-LLM
  thesis (grep: only a "climbing the value gradient" metaphor).
- Also briefly absent (lower priority): **structural plasticity** (spine turnover / synaptogenesis — the
  architecture is topologically fixed), and **excitability-based memory allocation** (CREB/excitability engram
  recruitment — distinct from the population reallocation in `predictions.py`).

### Capstone. The CORE itself learns biologically — grid cells under a non-backprop rule ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/emergent_grid_bio.py` (n=5). `emergence.py` shows periodic grid fields
  EMERGE when a recurrent cortex is trained on self-supervised path integration — but by **backprop** (the very
  thing #A1 says the cortex cannot do). This closes the loop: the same path-integration net is trained by
  **RFLO** (Murray 2019) — an **eligibility trace** (e-prop's temporal-credit primitive) × a learning signal
  through a **fixed random feedback** matrix (#A1's feedback alignment — no weight transport, no BPTT) — and the
  grid code still forms. (A) RFLO LEARNS path integration (place-loss **0.021 ≈ backprop 0.014**, ≪ untrained
  0.082) without weight transport; (B) the EMERGENT grid code appears (rate-map periodicity **0.53 ≈ backprop
  0.50**, **+0.09 ± 0.03** over the untrained floor; **76%** of units periodic vs 47%), never in the loss;
  (C) FALSIFIER — with the feedback **shuffled** every step the grid code falls to the untrained floor (0.45,
  **−0.09 ± 0.03**) even though its readout still fits, so it is the CONSISTENT feedback the forward weights
  align to (#A1), not any feedback, that grows grid cells. Honest scope: periodic multi-field cells, not a
  hexagonal lattice (gridness stays negative for backprop too — as in `emergence.py`). This moves the model from
  "biological rules bolted onto a backprop core" to **the core itself learning biologically.** See
  `results/FINDINGS.md`.

---

## Top recommendation

Tiers 1–2 are closed (#1 BTSP, #2 hexadirectional, #3 goal/reward, #4 social space, #5 neuromodulation); the
learning-substrate tier is closed — **#A1** (deep credit assignment without backprop), **#B3** (volatility-adaptive
meta-learning), **#B4** (astrocyte-gated slow plasticity) — and so is the **faithfulness capstone**: the *real*
recurrent grid cortex now learns its grid code under a non-backprop rule (RFLO = A1's feedback alignment +
e-prop), moving the model from "biological rules bolted onto a backprop core" to **the core itself learning
biologically.** **#B2** (Benna–Fusi multi-timescale synapse), **#C6** (representational drift — a label-free geometry read-out
survives geometry-preserving drift, even a full remap, but fails geometry-destroying drift at matched single-cell
drift; the circular first attempt was caught and rebuilt), and **#C7** (sleep triple-coupling — SO→spindle→ripple
nesting selects and times consolidation) are closed too. **The entire CPU register — every Tier-1/2/5 item — is
now closed.** The only remaining items are the **GPU/language capstones #8/#9** (a frozen LLM reads the
concept-grid / social-space maps, cortex-ON vs text-only-OFF), which need a T4 and live in `notebooks/`.
