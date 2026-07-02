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

### 1. Behavioral-timescale synaptic plasticity (BTSP) — one-shot place fields the *biological* way
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

### 2. A hexadirectional grid code over a **2-D conceptual** space (human abstract cognitive map)
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
- **Proposed experiment (emergence).** Define a 2-D conceptual space (two continuous features), drive the
  frozen velocity grid cortex along "trajectories" in it, and **measure the 6-fold rotational symmetry** of
  the population code's activity vs. movement direction (the hexadirectional signature) — emergent, never in
  the loss. Falsifier: a shuffled/curved-metric control should destroy the 6-fold signal. CPU. GPU follow-up:
  the LLM answers concept-space "which is closer?" from the grid-of-concepts code.

### 3. Vectorial **goal / reward** cells (direction-and-distance to a remembered goal)
- **Neuro basis.** Single neurons encode a **vector to a goal** (egocentric/allocentric direction + distance)
  — goal-vector cells in bats (Sarel, Finkelstein, Las, Ulanovsky 2017) and reward/goal coding + reward-biased
  place-field over-representation in rodents and humans (Gauthier & Tank 2018; Boccara 2019).
- **Model status: partial.** Homing to the **origin** exists (`agent_deadreckoning.py`, desert-ant homing),
  and object-vector cells point to an object — but there is no population explicitly tuned to
  **direction+distance to an arbitrary rewarded goal**, and no reward-driven field over-representation.
- **Proposed experiment (emergence).** Add goal-vector cells built from the existing grid/HD code (goal
  position − current position → egocentric vector). Train only to reach goals; **measure** the emergent
  vector tuning and whether place/grid resources **over-represent** the goal (a signature never trained). CPU.

### 4. Coding of **other agents** / social space
- **Neuro basis.** The hippocampus encodes **another individual's** position with dedicated "social place
  cells" (Danjo 2018; Omer, Las, Ulanovsky 2018 in bats), and humans map **social hierarchies** with the same
  grid/hippocampal machinery (Tavares 2015; Park, Miller 2021).
- **Model status: absent.** No representation of a second agent or social/relational-to-others map exists
  anywhere in `src/`.
- **Proposed experiment (emergence).** A second grid/place population tracks a *conspecific's* trajectory;
  in a task requiring the other's location, **measure** the emergence of other-agent place fields and a clean
  double dissociation (self-map vs. other-map lesions). CPU.

---

## Tier 2 — CPU, integration / harder

### 5. Neuromodulatory control of **encoding vs. retrieval** and **surprise-driven reset** — made faithful & emergent
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

## Top recommendation

Build **#2 (hexadirectional grid over a 2-D conceptual space)** and **#1 (BTSP one-shot plasticity)** next —
they are the two most cutting-edge, most human, and most measurable gaps, one on the "how the brain **works**"
axis (a grid code for *concepts*, the human cognitive-map signature) and one on the "how the brain **learns**"
axis (the real one-shot learning rule, replacing the episodic-store abstraction). Both are CPU, both reuse the
existing grid cortex, and both yield a clean emergent signature (6-fold symmetry; predictive field shift) that
is never put into the loss.
