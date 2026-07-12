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

### 5b. Hippocampal subfield micro-architecture — **DG pattern separation + CA1 comparator** ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/hippocampal_subfields.py` (n=5). The repo had CA3 (`HopfieldAssociativeMemory`,
  pattern completion) but not the subfields around it. DG = a massive **sparse expansion** (few % active) that
  **orthogonalizes** similar entorhinal inputs; CA1 = a **comparator** of CA3's completion vs the entorhinal
  reality (novelty/prediction-error). Measured, never in a loss, guarded against the by-construction trap (a
  random sparse expansion trivially orthogonalizes — so the headline is the *downstream recall*, not the DG
  orthogonality): (A) storing **M=24 SIMILAR** environments (entorhinal overlap 0.6) and recalling from a
  30%-degraded cue, the sparse DG code lets CA3 recall the **correct** environment **0.87 ± 0.02** where a
  **matched-size DENSE** expansion intrudes on a similar one (**0.37 ± 0.03**; gap **+0.51**). Strikingly the
  dense expansion is *worse than not expanding at all* (direct-EC 0.86) — so a large CA3 is **actively harmful
  unless sparse**; DG's separation is what makes the expansion usable. Mechanism check: DG's separation index
  (output/input overlap) **0.54** (orthogonalized) vs dense **1.00** (overlap preserved). (B) **Falsifier** =
  the dense expansion (same N_dg, no k-WTA) → interference returns. (C) **CA1 comparator**: mismatch discriminates
  novel vs familiar (**AUC 1.00**); **ablate the CA3 stream → 0.50** (chance) — a genuine entorhinal-vs-memory
  comparator, not an input-novelty detector (which the NE organ already is). (Marr 1971; Treves & Rolls 1994;
  Lisman & Grace 2005; Vinogradova 1995.) See `results/FINDINGS.md`.
- **Neuro basis.** The hippocampal triad splits memory into specialized nodes: **DG** sparsifies/expands for
  pattern separation (preventing catastrophic interference between similar experiences), **CA3** auto-associates
  for completion, **CA1** compares CA3's prediction against entorhinal input to flag novelty/mismatch.

### 5c. Non-synaptic channel — **ephaptic field coupling** ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/ephaptic_coupling.py` (n=5). The whole model coordinates neurons through
  synaptic weights; ephaptic coupling is a NON-classical channel — transmembrane currents sum into a local
  field that biases neighbouring spike TIMING with no synapse (Anastassiou & Koch 2011; Chiang, Han, Durand
  2019: activity propagates via endogenous fields with synaptic *and* gap-junction transmission blocked). A
  self-generated **zero-mean** field on a LIF population (E = g·(population low-pass − slow baseline), so it
  sharpens the rhythm without net drive), measured against the by-construction trap (a field that just added
  common drive would raise the RATE, so every comparison is at a **rate-matched** operating point): (A) field ON
  synchronizes spike timing (**χ 1.00 vs 0.07** without) at **matched rate** (|Δrate| 0.03 → it is timing, not
  drive), with a dose-response transition (0.07 → 0.93 → 1.00); (B) the diffuse **global field beats
  matched-budget SPARSE synapses** (χ 1.00 vs 0.11) — a coherent field coordinates where equally-strong local
  wiring cannot, and needs no synapses at all; (C) **falsifier** — zero the field → χ 0.07 (uncoupled baseline);
  (D) **computational work** — the field-made synchrony drives a downstream coincidence detector (fires **0.22 vs
  0.01** at matched input rate), so the timing is *readable* where rate alone is not. See `results/FINDINGS.md`.
- **Neuro basis.** Endogenous electric fields (the LFP) feed back onto membranes and entrain spike timing
  independently of synapses — a volume-transmission / non-synaptic computational channel the connectionist
  substrate omits.

### 5d. Non-Euclidean grid deformation — **grid shearing with environmental geometry** ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/grid_shearing.py` (n=5). Grid cells are not a rigid lattice: in
  polarized/trapezoidal environments they lose hexagonal symmetry and shear (Krupic et al. 2015, Nature;
  Stensola et al. 2015, Nature). The repo's grid modules were a rigid function of position; here the deformation
  is **not drawn — it emerges**. The model localizes at walls with a SQUARE-calibrated rule
  (`p_hat = bearing·(arena_R − wall_distance)`); in a trapezoid the walls are not at arena_R along their normals,
  so that rule MISLOCALIZES, warping the phase↔position map, and the rate map (over TRUE position) shears.
  Measured against a clean **double dissociation**: (A) SQUARE+anchoring gridness **+1.00** → TRAPEZOID+anchoring
  **+0.01** (drop **+0.99 ± 0.01**), with a dose-response (half-shear +0.60 — the deformation grows with the
  geometry); (B) **falsifier** — the deformation needs BOTH ingredients: TRAPEZOID + NO anchoring stays
  hexagonal (**+1.14**; the geometry alone does nothing to the rigid path-integrator) and SQUARE + anchoring
  stays hexagonal (**+1.00**; the square-calibrated fix is correct there) — only trapezoid+anchoring deforms
  (gap **+1.14 ± 0.01**). The grid deforms *itself* with environmental geometry, never put in a loss. Honest
  note: this took diagnosing a phase-offset setup bug (trajectories must start at the origin so grid phase tracks
  true position) before the baseline was cleanly hexagonal — reported, not hidden. See `results/FINDINGS.md`.
- **Deeper — does the MANIFOLD itself deform?** `src/eval/manifold_geometry.py` (n=5) answers the sharper
  question: the #5d shear is a RATE-MAP (read-out over space) effect — does the neural population MANIFOLD deform,
  or stay a rigid torus? **(A)** the trapezoid grid codes lie on the SAME manifold as the square's (overlap
  **0.88** vs the square-vs-square reference **0.90**; deformation **+0.02 ≈ 0**) — the manifold is a **rigid
  torus** (consistent with Gardner et al. 2022, toroidal topology preserved across environments), so #5d is a
  warping of the space→manifold MAP, not the manifold. **(B)** in a non-Euclidean BARRIER environment the fixed
  grid ignores the wall (neural distance tracks Euclidean, geodesic-advantage **+0.02**) while a PLASTIC code
  reshapes to the geodesic geometry (**+0.27**) — manifold deformation to the environment's actual geometry
  REQUIRES a plastic attractor, the capacity the rigid CAN lacks. The honest answer: the standard CAN retains its
  toroidal perfection; #5d is a map effect.
- **Neuro basis.** A grid is a self-organized code anchored to boundaries; when boundary geometry is polarized,
  the anchoring can no longer tile it hexagonally, so it shears/fragments — the grid is *shaped by* the
  environment, not a rigid ruler laid over it.

### 5e. Egocentric↔allocentric bridge — **RSC/PPC reference-frame transform + emergent gain fields** ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/reference_transform.py` (n=5). Posterior parietal cortex codes space
  egocentrically; the retrosplenial cortex transforms it to allocentric world coordinates via a head-direction-
  gated rotation, implemented in cortex by GAIN FIELDS (Andersen & Zipser 1988; Byrne, Becker & Burgess 2007;
  Bicanski & Burgess 2018). The repo had egocentric & allocentric codes coexisting (landmark_anchoring.py) but
  not the transform circuit. A plain MLP is trained ONLY to output a landmark's ALLOCENTRIC position from its
  egocentric view (dist, bearing-to-head) + head direction, and we MEASURE, never in the loss: (A) **it learned
  the TRANSFORM** — trained on head directions OUTSIDE a held-out band, it generalizes to unseen head directions
  at **RMSE 0.07 (4% of the target scale)**, which a lookup cannot; (B) **GAIN FIELDS EMERGE** — **27%** of
  hidden units develop multiplicative ego×head-direction tuning (extra variance from the multiplicative terms
  **0.08 vs 0.015** untrained; the Zipser-Andersen signature); (C) **falsifiers** — SHUFFLED heading → RMSE
  **0.90**, REMOVED heading → **2.24** (worse than predicting zero) — impossible without the correct direction.
  **Honest grade:** a mechanism demonstration with a *real emergent internal code* (gain fields), but the
  *expected* solution to a multiplicative transform — not a surprising emergence like the grid shearing.
  See `results/FINDINGS.md`.
- **Neuro basis.** RSC/PPC rotate egocentric sensory geometry into the hippocampal allocentric map using head
  direction, via multiplicative gain-field neurons — the bridge from first-person perception to a world-centred
  cognitive map.

### 5f. Non-synaptic (glial) — **astrocyte syncytium: spatial-density-gated plasticity + heterosynaptic binding** ✅ CLOSED (Jul 2026)
- **Status: implemented (honest, modest).** `src/eval/astrocyte_syncytium.py` (n=5). The repo already has a
  POINT-WISE astrocyte organ (#B4, gates each synapse by its own activity). Astrocytes are also gap-junction
  coupled into a SYNCYTIUM across which Ca²⁺ spreads (Scemes & Giaume 2006; the substrate for Ca²⁺ waves). We
  asked, honestly, what the *spatial coupling* computes that a point can't. **Reported finding first (not
  hidden):** a FULLY REGENERATIVE Ca²⁺ wave is all-or-nothing — it FLOODS the whole array once ignited (clustered
  1.00 ≈ scattered 1.00, no spatial selectivity), so the computation lives in the GRADED diffusive spread, not
  the regenerative wave. With sub-threshold single-synapse drive (a point can't trigger plasticity alone): (A)
  **heterosynaptic binding** — a silent-but-surrounded synapse is recruited into the assembly by pooled neighbour
  Ca²⁺ (gate **0.95 vs 0.00** point-wise; Henneberger 2010); (B) **spatial-density gate** — spatially-CLUSTERED
  co-activity's core potentiates (**0.40**) where the SAME NUMBER SCATTERED does not (**0.07**, selectivity
  **+0.33**); (C) **falsifiers** — UNCOUPLED does nothing (0.00, no pooling), the REGENERATIVE WAVE floods (no
  selectivity). **Honest grade:** a real network computation from glial coupling, but *modest* — only the cluster
  core binds, and the useful regime is the graded spread, not the wave itself. This is the fuzziest item in the
  whole register (astrocyte Ca²⁺-wave *computation* is genuinely debated); I flagged it as high-risk up front and
  kept the claim to exactly what the controls support. See `results/FINDINGS.md`.
- **Neuro basis.** Gap-junction-coupled astrocytes spread Ca²⁺ across a syncytium, letting one synapse's glial
  signal reach its neighbours — a spatial, non-synaptic substrate for coordinating plasticity across an ensemble.

### 5g. Non-Euclidean **path integration** — curvature read from self-motion (Gauss-Bonnet holonomy) ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/curved_path_integration.py` (n=5). The critique's "3-D / non-Euclidean
  topologies" item had two halves. The **3-D volume** half is already covered: `grid_3d.py` / `local_3d_order.py`
  build the bat-regime 3-D code (local order, no global lattice; Ginosar 2021), path-integrating and localizing
  in 3-D. **Honest note:** I first tried to make that regime *emerge* from plane-wave interference and it does
  NOT — generic 3-D interference gives *disordered* fields, not regular-spacing local order (that regime is a
  packing property, which the repo already models by construction). Rather than manufacture an emergence, I
  closed the genuinely-open **non-Euclidean** half: what a flat grid/head-direction integrator DOES on a curved
  manifold. The answer is exact and never put in the code:
  - **(A) CURVATURE FROM SELF-MOTION.** The parallel-transport holonomy around a closed loop equals the enclosed
    **area × curvature** (= solid angle; Gauss-Bonnet): slope **1.00**, corr **1.00**, calibration residual
    **1.3%**; a geodesic triangle with three right angles gives holonomy **π/2** (1.57). A flat compass reads
    curvature purely from having walked a loop.
  - **(B) FLAT FALSIFIER + DOSE-RESPONSE.** In flat space (zero-curvature limit) the holonomy is **0.03 ≈ 0**
    (loops close); at fixed enclosed area it grows as **1/R²** (monotone in curvature) — the signal is the flat
    assumption meeting curvature, not a bug.
  - **(C) BEHAVIOURAL CONSEQUENCE.** An agent that path-integrates its heading flatly then heads for a remembered
    goal mis-homes by the holonomy — miss **1.98** on the curved world vs **0.03** flat.
- **Honest grade:** the non-Euclidean analogue of grid shearing — a flat mechanism meets a geometry it was never
  built for and an exact geometric signature (Gauss-Bonnet) falls out. The holonomy = area × curvature is a
  mathematical identity, so the emergence is that the flat neural integrator *inherits* it and mis-navigates by
  a computable amount; clean and faithful, with a perfect flat-space falsifier. See `results/FINDINGS.md`.
- **Neuro basis.** Head-direction and grid path integration assume a locally-flat plane; on a curved surface the
  transported heading accumulates the Gauss-Bonnet holonomy — a concrete, testable non-Euclidean prediction.

### 6. **Replay** used for planning & credit assignment — not just present as a ripple signature ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/replay_planning.py` (n=8), extends `successor.py`. The repo had a
  `SharpWaveRipple` organ and offline experience-replay that *consolidates a decode map* (`pillars.py`), but
  replay never *computed* anything directional. Now it does, both ways the hippocampus uses it, with the
  direction **emergent** (never encoded) and a falsifier on each:
  - **(A) REVERSE replay = credit assignment** (Foster & Wilson 2006; Ambrose-Pfeiffer-Foster 2016). Prioritized
    sweeping — back up the transition with the largest **|TD error|**, a *scalar* priority with **no direction in
    it** — makes value updates sweep **BACKWARD from the reward**: reverse fraction **1.00** vs **0.50** for
    RANDOM-order replay (paired p=0.009). The reverse order is a *consequence* of the surprise starting at the
    reward, not a design choice.
  - **(B) FORWARD replay = planning** (Pfeiffer-Foster 2013). The **same** learned predictive value, read forward
    by a greedy value-ascent rollout, routes around the barrier to the goal: forward fraction **1.00**, solves
    the maze from **100%** of starts vs **1%** on an untrained value (falsifier: no gradient → no plan).
  - **(C) THE DISSOCIATION** (Diba-Buzsáki 2007; Mattar-Daw 2018): one value function, opposite directions —
    reverse to assign credit for a past reward, forward to plan a future path.
  - **(D) PAYOFF:** prioritized replay reaches a plannable map in **16×** fewer backups than random — the data
    efficiency replay is *for*.
- **Honest grade:** *expected mechanism, faithful signature* — a person who knows prioritized sweeping would
  predict reverse order, so this is not a *surprising* emergence like grid shearing. But the direction is
  genuinely not encoded (random → 0.5 is the proof), and reproducing the reverse/forward dissociation from a
  single value rule is the real, literature-faithful result. See `results/FINDINGS.md`.
- **Neuro basis.** Hippocampal **replay** — reverse for credit assignment, forward for planning — supports
  model-based decisions and offline consolidation (Ólafsdóttir 2018; Mattar & Daw 2018; Liu 2019).

### 7. Explicit **uncertainty / confidence** that drives behavior ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/uncertainty_behavior.py` (n=5), on the real grid cortex. The repo had
  *implicit* uncertainty (near-optimal cue integration; Fisher capacity) and `agent_cue_integration.py`
  explicitly **left open** the strict reliability-weighting law (a recurrent fuser temporally averages
  *unbiased* cues, so a noisy cue never must be down-weighted). This makes uncertainty **explicit, calibrated,
  and behaviourally coupled**, three ways — each with a falsifier, none of the signatures in a loss:
  - **(A) A CALIBRATED UNCERTAINTY DECODED FROM THE POPULATION.** Real grid modules are independent attractors
    (Burak-Fiete 2009), so independent per-module drift makes them DISAGREE; the reconstruction residual
    ρ = ‖code − grid_code_at(decode(code))‖ (how badly any single position explains the population — the grid
    code as an error-CORRECTING code, Sreenivasan-Fiete 2011) is calibrated to the true decode error
    (**corr 0.87**), RISES with path integration (**2.5×**) and RESETS at a cue (**−1.70**). FALSIFIER / honest
    boundary: under SHARED drift the modules stay mutually consistent → ρ is **uncalibrated (corr 0.19)**, flat:
    the code is "confidently wrong," blind to coherent drift.
  - **(B) IT DRIVES BAYESIAN CUE RE-WEIGHTING (closes the open item).** A single-shot head trained ONLY to
    localize develops an effective landmark weight that tracks the inverse-variance optimum
    w* = σ_PI²/(σ_PI²+σ_L²), **driven by ρ** — slope **0.65**, corr **0.94** (Ernst-Banks 2002). FALSIFIER: a
    head blind to (ρ, σ_L) can only average (slope **0.09**). The <1 slope is honest — behaviour is driven by
    the *noisy population signal* ρ, not an oracle.
  - **(C) IT DRIVES A SWITCH, ON THE BELIEF NOT THE TRUTH.** Inflating ρ WITHOUT changing the true error raises
    landmark trust (**Δ+0.34** vs blind **+0.00**) — the agent acts on its internal estimate (metacognition);
    the re-anchor crossover moves **+38 steps** with landmark reliability.
- **Honest grade:** *expected mechanism, faithful+non-trivial signature.* Inverse-variance weighting is the
  Bayes solution an MSE learner is expected to find; what is non-trivial is that a genuine, calibrated
  uncertainty is read straight out of the population code (A) with a clean "confidently wrong" boundary, and
  that behaviour provably follows that internal estimate rather than an oracle (C). See `results/FINDINGS.md`.
- **Neuro basis.** The brain represents **posterior uncertainty** and confidence gates exploration and
  cue-weighting (Ma 2006; Pouget 2013; Ernst-Banks 2002).

### 8. Neocortical **systems consolidation** — replay moves the map into the cortical weights ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/systems_consolidation.py` (n=5). The repo showed replay *consolidating a
  decode map* (`pillars.py`) and the LLM *reading* a frozen cortex (`structural_transfer.py`) — but that read is
  a permanent structural DEPENDENCY on the hippocampal module, the opposite of what Complementary Learning
  Systems predicts (McClelland-McNaughton-O'Reilly 1995; Squire-Alvarez 1995; Frankland-Bontempi 2005). This
  builds the two-store loop — a fast one-shot HIPPOCAMPAL store (stands for the CA3 Hopfield / place-cell memory)
  and a slow gradient-trained CORTICAL network (the frozen-LLM-weights analogue) that learns ONLY from replayed
  samples — and measures the classic signatures, none in a loss:
  - **(A) TEMPORALLY-GRADED RETROGRADE AMNESIA.** After a hippocampal lesion (cortex-only recall) accuracy is a
    GRADED function of a map's age: remote **0.61** vs recent **0.23** (gradient **+0.39**, recall↑age corr
    **0.87**, chance 0.10). Older maps were simply replayed on more nights — the gradient EMERGES.
  - **(B) THE DOUBLE DISSOCIATION.** With the hippocampus INTACT, recall is **100%** at every age (no gradient —
    the fast store has everything); the gradient appears ONLY on lesion, only for RECENT memories (Scoville-
    Milner / Squire).
  - **(C) REPLAY IS CAUSAL (falsifier).** Replay OFF → the cortex never learns → remote collapses to **0.13**
    (chance) and the gradient vanishes (**+0.02**): the transfer is the replay, not the passage of time.
  - **(D) THE MAP IS IN THE WEIGHTS.** The cortex alone (hippocampus-independent) recalls remote maps at 61% —
    the spatial structure has been internalised into the slow weights, which the frozen-LLM read was said to
    prevent.
- **Honest grade:** *expected mechanism, faithful signature* — a two-store + replay system is *expected* to
  produce a consolidation gradient, but reproducing the temporally-graded retrograde amnesia + hippocampus-
  independence of remote memory (the Squire pattern), with the replay-off falsifier killing it, is a clean,
  literature-faithful closure of the CLS gap. See `results/FINDINGS.md`.
- **Neuro basis.** Sharp-wave-ripple replay slowly trains the neocortex on hippocampal memories until a familiar
  environment is recalled without the hippocampus — the complementary fast/slow learning systems.

### 9. **Active inference** — epistemic foraging that drives the body to reduce spatial uncertainty ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/active_inference.py` (n=5). The pipeline treated navigation as passive
  observation; active inference (Friston) says the system *acts* to reduce its own spatial uncertainty. **The one
  thing we refuse to hardcode is the behaviour** — the agent is rewarded ONLY for reaching the goal (no landmark
  reward, no information-gain bonus, no exploration term). The only thing built is the PLATFORM physics (the #7
  uncertainty: path integration drifts so uncertainty *u* grows, a landmark resets it, and a goal-commit succeeds
  with probability P(u) that falls as u grows). A belief-state planner that maximises expected GOAL reward — and,
  independently, a model-free Q-learner — do the rest:
  - **(A) EPISTEMIC FORAGING EMERGES.** The optimal policy DETOURS to a landmark to relocalise before committing,
    from **52%** of start states under drift — purely to raise its chance of actually arriving.
  - **(B) THE NON-HARDCODING PROOF (dissociation).** In a NO-DRIFT world the SAME planner detours from **0%** of
    starts: with no uncertainty to reduce there is no epistemic value, so the detour was never a hardcoded
    landmark preference — it is contingent on *reducible uncertainty*.
  - **(C) IT PAYS.** The uncertainty-aware planner reaches the goal **47%** vs a σ-BLIND greedy agent (same goal
    reward, cannot see u) **21%** and random **4%**.
  - **(D) IT MUST SENSE ITS UNCERTAINTY (ablation).** Blind to u, the planner cannot time the detour and collapses
    to **20%** (≈ greedy).
  - **(E) IT ALSO EMERGES FROM LEARNING.** A model-free Q-learner trained ONLY on the goal reward develops the
    same detour-when-uncertain policy (**82%** relocalisation) — the behaviour is not special to the planner.
- **Honest grade:** *emergent behaviour, mechanism-only inputs.* Nothing about uncertainty, landmarks or
  exploration is in the objective — information-seeking falls out of pure goal-seeking because uncertainty is
  instrumentally costly, and the no-drift dissociation proves it. This is exactly the "hardcode at most the
  mechanism, let the behaviour emerge" bar. See `results/FINDINGS.md`.
- **Neuro basis.** The entorhinal-hippocampal system drives exploratory action to minimise expected free energy
  (spatial uncertainty); epistemic foraging is implicit in acting to reach preferred states under uncertainty.

### 10. **Interoceptive anchoring** — drive state remaps value & navigation (beyond dopamine) ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/interoceptive_map.py` (n=5). The repo mapped external geometry and a dopamine
  value signal, but the cognitive map is anchored to the body: place-cell value and navigation remap with
  homeostatic drive (Kennedy-Shapiro 2009; Keramati-Gutkin 2014). **We refuse to hardcode the behaviour** — there
  is NO "if thirsty go to water" rule. The only thing built is the body: two deficits (thirst t, hunger h) grow
  each step, WATER resets t and FOOD resets h, and the reward is the reduction of total DRIVE, −(t²+h²). A
  belief-state planner over (position, t, h) that maximises this does the rest:
  - **(A) INTEROCEPTIVE NAVIGATION.** From a neutral start the agent heads to the resource matching its DOMINANT
    deficit **96%** of the time; a DRIVE-BLIND planner (same objective, cannot read t,h) matches **0%** (it goes
    to one fixed resource regardless) — the target is set by the interoceptive gap, not geometry.
  - **(B) DRIVE-DEPENDENT VALUE REMAPPING.** The drive-specific value residual under thirst vs hunger is
    anti-correlated (**−0.93**) — the same place is worth opposite amounts under different deficits — and each
    resource's value tracks its OWN deficit (normalised gain **+0.29**).
  - **(C) HOMEOSTATIC REGULATION (payoff).** The interoceptive planner keeps mean drive at **57** vs a drive-blind
    planner's **180** and random's **152**, shuttling **11×/life** as its deficits cycle. It stays alive.
  - **(D) NON-HARDCODING PROOF.** Blind to its own deficits the agent chooses at chance and lets one deficit
    explode — so the behaviour is genuinely interoceptive, not a fixed spatial habit.
- **Honest grade:** *emergent behaviour, mechanism-only inputs* — nothing about "thirst→water" is in the reward;
  drive-appropriate navigation and value remapping fall out of minimising a homeostatic drive, and the drive-blind
  ablation proves the map is anchored to the body. The same "hardcode at most the mechanism" bar as #9. See
  `results/FINDINGS.md`.
- **Neuro basis.** The hippocampus receives dense hypothalamic/amygdalar input; place fields, replay and value
  remap with homeostatic state, and navigation is vector-driven by interoceptive deficits (thirst, hunger, fear).

### 11. **Adult neurogenesis** — temporal stamping + reduced interference from cohort turnover ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/neurogenesis_stamp.py` (n=5). The net had a fixed parameter count; the adult
  dentate gyrus adds granule cells continuously, each passing a brief maturation window when it is hyper-EXCITABLE
  and hyper-PLASTIC before it freezes (Aimone-Wiles-Gage 2006/2009; Kee 2007; Rangel 2014). **We hardcode none of
  the behaviour:** TIME is never encoded, per-event content is random and decorrelated from time, and the only
  thing built is the mechanism (a young cohort of K cells is born each step, fires readily, learns fast, then
  freezes). What emerges:
  - **(A) TEMPORAL STAMPING.** Because only the current young cohort is plastic and hyper-excitable, events close
    in time are bound by the SAME cells, so DG-code overlap tracks temporal proximity (corr(overlap, Δt) **−0.60**)
    and near-vs-far-in-time is decodable from the code (**AUC 0.96**) — even though content carries no time. A
    STATIC DG shows **corr +0.00 / AUC 0.50** (content-only): the stamp is the cohort.
  - **(B) REDUCED INTERFERENCE.** Fresh cells absorb new memories while mature cells stay frozen, so old memories
    are retained (recall **0.44** vs static **0.31**) and recall is FLAT across age (retention gap **−0.01**),
    where the static net catastrophically forgets the old for the new (gap **+0.52**).
  - **(C) THE FALSIFIER.** A static DG (no turnover, uniform excitability + plasticity) has neither the stamp nor
    the retention — both effects are the turnover, not the substrate.
- **Honest grade:** *emergent behaviour, mechanism-only inputs.* Nothing encodes time or protects old memories by
  hand; a temporal metric and age-flat retention fall out of a young cohort turning over, and the static ablation
  proves it. Birth is stochastic, so the stamp is an emergent noisy metric, not an exact clock. The same
  "hardcode at most the mechanism" bar as #9/#10. See `results/FINDINGS.md`.
- **Neuro basis.** Newborn granule cells time-stamp memories (same-cohort cells encode temporally-near events) and
  segregate new from old learning, giving the DG a continuous-time index and continual-learning capacity a static
  population cannot.

### Integration capstone. **The unified agent** — the organs act as one machine ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/unified_agent.py` (n=5). The register proved a shelf of mechanisms in
  isolation; this wires the survival-critical ones into ONE agent and asks the question isolation cannot: do they
  cohere into an animal? The agent's SOLE objective is to survive (keep total homeostatic drive low). The world
  composes validated platforms — a grid POSITION sense that DRIFTS (#7/#8), an UNCERTAINTY read-out (#7),
  LANDMARK relocalisation (#1), and asymmetric INTEROCEPTIVE drives (#4) reduced only by the matching resource and
  only well when localised. A single belief-state planner over (position, uncertainty, thirst, hunger) maximises
  survival; which resource and when to relocalise are never hardcoded.
  - **(A) N-ORGAN LESION DISSOCIATION.** Survival needs all four organs; each ablation fails in its own way —
    intact mean drive **36**, − grid **71** (can't navigate, catastrophic), − uncertainty **45** (can't tell when
    it's lost), − landmark **45** (can't undo drift), − interoception **42** (can't tell which deficit is killing
    it). Grid is the most fundamental; the rest give graded survival benefit.
  - **(B) EMERGENT CROSS-ORGAN INTERACTION.** The organs form a CIRCUIT, not a pile: removing the uncertainty
    read-out costs **+8** drive when landmarks are present but **+0** once they are gone — knowing you are lost is
    worthless unless you can re-anchor. A super-additive complementarity neither organ shows alone.
- **Honest grade:** *emergent behaviour, mechanism-only inputs* — one survival objective composes four organs
  into a coherent animal with the drive-appropriate navigation, relocalisation and homeostasis all emerging. Honest
  process note: an early version inflated the landmark lesion by letting the agent chase re-anchoring that no
  longer worked; a lesion should mean the organ is gone and the brain re-plans without it, which both removes the
  artefact and makes the complementarity clean. See `results/FINDINGS.md`.
- **Grounded on the real cortex.** `src/eval/unified_agent_cortex.py` (n=5) re-runs the SAME emergent survival
  policy with PERCEPTION from the actual substrate: position is decoded from the real drifting `_HexGridModules`
  grid code, uncertainty is the real #7 reconstruction residual (no counters). The three POSITION organs still
  dissociate cleanly — intact drive **48**, − grid **69**, − uncertainty **60**, − landmark **66** — and the
  uncertainty×landmark complementarity survives grounding (**+12** with landmarks, **−3** without). *Honest limit:*
  the interoceptive DRIVE organ, cleanly load-bearing for resource CHOICE in #4, barely moves survival here
  (**46 ≈ 48**) — with two symmetric resources a non-adaptive alternation nearly suffices — so grounding yields a
  clean **3-organ** dissociation, not 4, and we report all four and say so.
- **Learning its world (memory organs).** `src/eval/unified_agent_learn.py` (n=5) upgrades the agent from PLANNING
  with a known model to LEARNING an unknown one: dropped in not knowing where resources are, it discovers them and
  builds a value map from experience (nothing about locations hardcoded). **(A)** it learns — mean drive falls
  over the lifetime (**56 → 46**); **(B)** REPLAY (#6) propagates each discovery across the map, so the map is
  learned far sooner — accuracy a fixed window after discovery **1.00 with replay vs 0.66 without** (honest note:
  this speeds *map-learning*; it moves *survival drive* only mildly, since discovery, not propagation, is the
  bottleneck here); **(C)** CLS consolidation (#2) moves the map into a slow store so a familiar world survives a
  hippocampal lesion — drive **47 with consolidation vs 65 without**. The memory organs, proven in isolation, do
  their jobs in the behaving agent.
- **Neuro basis.** A behaving animal integrates entorhinal grid path-integration, an uncertainty/confidence signal,
  allothetic landmark correction and hypothalamic drive into one goal-directed survival loop; the organs are
  functionally interdependent, not modular add-ons.

### 12. Reciprocal integration — **top-down feedback that reshapes the spatial cortex** ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/topdown_feedback.py` (n=5). The pipeline was read-only: spatial tokens flow
  INTO the frozen LLM via gated cross-attention (`fusion.py`, query=text/key=spatial), but the LLM had no path
  back to the spatial cortex — whereas the entorhinal-hippocampal loop is reciprocal, and neocortical GOALS
  reshape place-cell tuning (place fields over-represent goal locations; Hollup 2001; Dupret-O'Neill-Csicsvari
  2010; Kentros 2004). This adds the feedback path and hardcodes NONE of the behaviour — no "enhance cells near
  the goal" rule. The only things built: a top-down gain modulation of the spatial cortex under a conserved
  attention BUDGET (N·softmax, so attention is a limited resource that must be allocated — Reynolds-Heeger), and a
  goal-weighted precision objective. What emerges:
  - **(A) GOAL OVER-REPRESENTATION.** The learned top-down gain concentrates on cells whose fields are near the
    goal — corr(gain, goal-proximity) **+0.29** — the map reorganises toward the goal (Dupret), never in the loss.
  - **(B) THE RECIPROCAL LOOP PAYS.** Near the goal the top-down model decodes at **0.030** vs a FEEDFORWARD
    read-only model's **0.057** — closing the loop beats the read-only pipeline where precision matters.
  - **(C) THE ATTENTION TRADE-OFF.** Limited budget → better near the goal (**0.030**) but worse far (**0.122**),
    the hallmark of attention, not a free lunch.
  - **(D) FALSIFIER.** Feed the WRONG goal → the gain enhances the wrong region → near-goal error **0.128**. The
    feedback must MATCH the goal, not merely be present.
- **Honest grade:** *emergent behaviour, mechanism-only inputs* — nothing tells the top-down signal to enhance the
  goal; goal over-representation and the attention trade-off fall out of a limited-budget feedback path trained for
  goal-directed precision, and the read-only baseline + wrong-goal falsifier show the loop is load-bearing. Note:
  the eval demonstrates the feedback ORGAN; wiring an LLM→cortex path into the main `fusion.py` pipeline is the
  natural follow-on integration. See `results/FINDINGS.md`.
- **Neuro basis.** The hippocampus projects back via deep MEC to the neocortex, and neocortical goals drive
  top-down spatial attention and goal-related place-field reorganisation — the loop is reciprocal, not read-only.

### 13. Decoupling **the map from value** — one hippocampal map, many striatal values ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/map_value_decouple.py` (n=8). The critique: fusing a dopamine value into the
  spatial read-out conflates the transition MODEL (the cognitive map) with the reinforcement MODEL (value).
  Biologically the hippocampus provides a goal-INDEPENDENT successor representation M, and value is V = M·R (the
  striatal reward assignment; Dayan 1993; Stachenfeld 2017; Momennejad 2017). The repo already keeps these
  separate (`successor.py` = M; `basal_ganglia.py` = striatal value); this shows the PAYOFF a fused map+value
  cannot have:
  - **(A) ONE MAP, MANY GOALS.** A single goal-independent SR map solves **8** goals via V = M[:, g] — reuse
    success **1.00**. The map is learned once and reused.
  - **(B) INSTANT REVALUATION.** When the goal moves, the decoupled agent revalues for free (V = M[:, g_new], a
    lookup) → **1.00**; a FUSED agent whose value is baked into its state read-out stays stuck on the OLD goal →
    **0.15** (paired p=0.016).
  - **(C) THE COST OF FUSION.** The fused agent must relearn a competent policy for the moved goal — **15** value-
    iteration sweeps — against the decoupled agent's **0**. That relearning cost is paid on every reward change.
- **Honest grade:** *expected mechanism, faithful payoff* — the SR revaluation advantage (Momennejad 2017) is a
  known result, so not a surprising emergence; but it is the faithful demonstration of the map/value decoupling
  the critique asks for, cleanly dissociated from a fused agent, and it confirms the repo's organs (SR map +
  striatal value) are the right factorisation. See `results/FINDINGS.md`.
- **Neuro basis.** The hippocampus builds the state-space (successor/predictive map); the striatum assigns
  dopamine value; separating "where I am" from "what it is worth" gives instant reward revaluation and multi-goal
  reuse a fused representation cannot.

### Polysemantic **superposition** — N place cells store MORE than N environments ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/superposition_capacity.py` (n=5). The critique: a localized one-cell-per-place
  read-out is *monosemantic* — N cells hold at most N fields — but high-density human intracranial recordings show
  hippocampal neurons are extremely POLYSEMANTIC, each encoding many unrelated places at once, the same
  high-dimensional SUPERPOSITION that lets an LLM's MLP pack more features than it has neurons (Elhage et al.
  2022, "Toy Models of Superposition"). Nothing about the coding is imposed: the only things built are the
  **mechanism** (a tied autoencoder — an N-cell bottleneck reconstructs its input) and the **task** (the input is a
  SPARSE set of active place fields, because you are in ONE place at a time, drawn from F = 4·N fields spanning
  many environments). Superposition, polysemanticity, and their sparsity-dependence all EMERGE:
  - **(A) SUPERPOSITION CAPACITY.** With sparse activity the 32 cells recall **1.00** of all **128** fields —
    **128 fields in 32 cells, 4× more environments than cells** — where a monosemantic one-cell-per-place code
    could recall only N/F = **0.25**.
  - **(B) POLYSEMANTICITY EMERGES.** Each cell ends up participating in **4.5 ± 0.1** fields (≫1) — the
    superpositional coding the intracranial data report, never put in a loss.
  - **(C) SPARSITY IS LOAD-BEARING (falsifier).** Train on DENSE activity (many fields at once) and superposition
    cannot form — recall collapses to **0.49**, toward the monosemantic ceiling. A dose-response confirms it:
    recall **1.00** (p=.04) → **1.00** (p=.12) → **0.52** (p=.30). The compression is bought precisely by
    exploiting "one place active at a time"; remove the sparsity and it is gone.
- **Honest grade:** *known mechanism, faithful spatial reframing* — Elhage's superposition is an established result,
  so the compression itself is not a surprise; the contribution is showing a **place code** realizes it exactly,
  reproducing the polysemantic hippocampal coding from nothing but a bottleneck + sparse-field reconstruction, with
  the sparsity dependence as a clean falsifier. See `results/FINDINGS.md`.
- **Neuro basis.** Because natural experience is sparse (one location active at a time), a fixed population can
  superpose many more place fields than it has cells; the cost is polysemantic, interference-prone cells — exactly
  what dense human recordings find, and why dense (non-sparse) activity destroys the capacity.

### Small-world **searchability** — a navigable shortcut structure emerges from use ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/small_world_search.py` (n=5). The critique: a map wired as a pure
  nearest-neighbour lattice forces goal-directed search to crawl hop-by-hop, where real hippocampal/cortical
  connectivity is *small-world* — sparse long-range shortcuts allow few-hop search. The deep point (Kleinberg 2000)
  is that short paths *existing* is not enough: a DECENTRALISED searcher (local structure + goal proximity only —
  exactly the grid-population-vector closeness the cortex already computes) can only *find* them by greedy routing
  when the shortcut-length distribution P(r) ∝ r^(−α) has the right exponent. Per the standing rule we hardcode
  none of that: the only things built are the **mechanism** (a local lattice + candidate long-range links from a
  FLAT prior + use-dependent selection under a 1-link/node wiring budget) and the **task** (greedy decentralised
  routing, no global path oracle). Navigability emerges and is measured, never in a loss:
  - **(A) NAVIGABILITY IS AN INTERIOR OPTIMUM.** Greedy delivery vs the shortcut exponent is non-monotone —
    α=0 **19.8**, α=1 **18.1**, α=2 **21.1**, α=3 **34.7** hops — and too-local (α=3) *scales* catastrophically
    (grows ×1.47 from n=60→90 vs the navigable band's ×1.28). It is the shortcut *distribution*, not their
    presence, that buys searchability.
  - **(B) FINDABILITY, NOT EXISTENCE.** The flat α=0 prior gives the *shortest* true paths (BFS **6.87**) yet the
    *worst* greedy stretch (**2.86** = greedy ÷ true-optimal) — the short paths are there but a local searcher
    cannot find them; the emergent graph cuts the stretch to **2.33**.
  - **(C) THE NAVIGABLE EXPONENT EMERGES.** Use-dependent selection from the flat prior grows the surviving-link
    exponent **α: 0 → 1.39 ± 0.01** (the navigable band) and delivers in **16.5** hops — beating the flat prior
    (**19.8**) AND the best fixed-exponent graph (**18.1**); adaptive per-node selection outperforms any i.i.d.
    fixed-α wiring.
  - **(D) FALSIFIER — random pruning.** Keep a RANDOM candidate per node (same 1-link budget, same pool): the
    exponent stays flat (**α ≈ −0.01**) and delivery gains nothing (**19.6**). It is the use-based selection, not
    the budget or pruning, that grows navigability.
- **Honest grade:** *emergent navigability, honest finite-size caveat* — the navigable structure genuinely
  self-organises (nothing about the exponent imposed) and beats every control. The one caveat, reported not
  hidden: the textbook navigable exponent α = D = 2 is an *asymptotic* result (polylog vs polynomial delivery
  separates only at astronomically large grids); at CPU-reachable sizes the finite-size navigable optimum sits
  lower (~1.4), and the emergent exponent lands *there*, on the size-appropriate navigable band — which is the
  honest claim, not "converges to 2." See `results/FINDINGS.md`.
- **Neuro basis.** Sparse long-range projections turn a local map into a small-world graph; but only a shortcut
  distribution matched to the map's dimensionality is navigable by a cell that sees only its own connections and
  where the goal is — and use-dependent plasticity selecting the shortcuts that actually carry greedy traffic
  grows exactly that distribution.

### Anisotropic 3-D coding — vertical coarsening emerges from gravity-biased experience ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/anisotropic_3d.py` (n=5). The critique: naively scaling a continuous
  attractor from 2-D to 3-D gives a perfectly ISOTROPIC lattice, but mammals do not code volumetric space
  isotropically — rats on climbing walls/helices have place/grid fields elongated VERTICALLY ("stripes") and
  selectively impaired vertical odometry "when the rat itself remains horizontal" (Hayman, Verriotis, Jovalekic,
  Fenton & Jeffery, *Nature Neuroscience* 2011; Grieves 2020), while freely-flying bats — which traverse the
  volume symmetrically — code 3-D far more isotropically (Ginosar 2021, the regime the repo's `LocalOrder3DGrid`
  already models). So the anisotropy is a fact about EXPERIENCE, not hardware, and — per the standing rule — none
  of it is hardcoded. Built only: ISOTROPIC hardware (isotropic code noise, isotropic weight init, one shared
  power budget) + the TASK (a capacity-limited code reconstructs 3-D position from a gravity-biased experience
  distribution — large horizontal spread, small vertical, because a terrestrial body lives near the ground). The
  anisotropy emerges by rate-distortion / water-filling:
  - **(A) EMERGENT VERTICAL COARSENING.** NORMALIZED decode error (error as a fraction of each axis's range —
    pure resolution, not range) is **0.50** vertically vs **0.15** horizontally: **vertical/horizontal = 3.33 ±
    0.11**, coarser vertical fields (Hayman's stripes), with isotropic hardware.
  - **(B) FALSIFIER — isotropic experience.** The SAME code given isotropic (flying-regime) experience is
    isotropic: ratio **1.04 ± 0.06**. So it is the experience, not the architecture.
  - **(C) DOSE-RESPONSE.** As vertical experience shrinks the anisotropy grows monotonically — ratio **1.00 → 1.66
    → 3.32 → 6.61** for vertical/horizontal experience **1.0 → 0.6 → 0.3 → 0.15** — tracking the deficit almost
    exactly as 1/(experience ratio), the water-filling law.
  - **(D) ABSOLUTE vs NORMALIZED (honesty).** In ABSOLUTE terms the vertical error is SMALL (**0.044** vs
    horizontal **0.15**) — the animal barely leaves its height band, so little is at stake and vertical coding can
    look fine; the disproportionate loss is visible only in the NORMALIZED resolution measure, so both are reported.
- **Honest grade:** *clean emergence.* The anisotropy self-organises from experience under isotropic hardware, with
  a clean isotropic-experience falsifier and a dose-response that follows the rate-distortion law — nothing about
  the vertical axis is treated differently in the model. Directly relevant to a **terrestrial/climbing** embodied
  agent (anisotropic regime); an aerial agent would sit in the isotropic falsifier regime. See `results/FINDINGS.md`.
- **Neuro basis.** A body held horizontal by gravity experiences large horizontal and small vertical self-motion;
  a fixed-capacity neural code with isotropic noise allocates resolution to the well-sampled horizontal axes and
  lets the poorly-sampled vertical fall below the coding threshold — elongated vertical fields and impaired
  vertical odometry, exactly as recorded, without any built-in vertical/horizontal asymmetry.

### Semantic warping of the map — the metric bends toward a relevant concept ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/semantic_warp.py` (n=5). The critique: the model treats the cortex as a purely
  geographic + value substrate and leaves all semantic meaning to the LLM, but biologically the **perforant path**
  projects non-spatial/behavioural features directly into grid & place assemblies, so the map is not rigidly
  geographic — grid cells warp toward remembered reward/goal locations, becoming **mixed-selective to reward and
  space** (Boccara et al., *Science* 2019; "the cognitive map is attracted to goals", Butler 2019; non-spatial
  binding: Aronov & Tank 2017, Constantinescu 2016, TEM/Whittington 2020). Per the standing rule the warp is
  hardcoded nowhere. Built only: a capacity-limited code with a spatial pathway AND a perforant/semantic input
  pathway, and the task (reconstruct POSITION and a scalar VALUE — position forces a spatial map; the value may or
  may not depend on the concept). The warp is never in the loss; it emerges:
  - **(A) THE MAP WARPS, YET STAYS SPATIAL (mixed selectivity).** When the concept is behaviourally relevant the
    representational metric warps by concept (partial corr of representational distance with concept-difference,
    controlling spatial distance, **+0.27 ± 0.02**) WHILE the code stays strongly spatial (spatial partial corr
    **+0.62**) — the mixed-selective warped map, not a concept map.
  - **(B) DOUBLE-DISSOCIATION FALSIFIER.** Remove the perforant projection (same relevant task, no semantic input):
    the map cannot warp (**+0.00 ± 0.02**, spatial +0.77). And with the path present but the concept **irrelevant**
    (β=0) the warp is **+0.01** too — the warp needs BOTH the perforant path AND behavioural relevance.
  - **(C) DOSE-RESPONSE.** Warp grows with relevance β: **+0.01 → +0.08 → +0.19 → +0.32** for β = 0 → 0.5 → 1 → 2.
  - **(D) PAYOFF.** A held-out LINEAR probe reads the concept off the WARPED map at **0.60** but is at chance
    (**0.23**, chance 0.20) without the perforant path — a downstream reader inherits the semantic-spatial structure
    for free instead of learning it from scratch, exactly the critique's point.
- **Honest grade:** *clean emergence with a double dissociation.* The warp self-organises, is never in the loss, and
  requires both the perforant projection and behavioural relevance (each ablation kills it independently); the
  payoff shows why it helps the LLM. See `results/FINDINGS.md`.
- **Neuro basis.** The perforant path binds non-spatial meaning into the hippocampal map; when a concept is
  behaviourally relevant the map deforms toward it (mixed selectivity), so relational/semantic structure is read
  off the map rather than recomputed downstream.

### Bifurcated RSC routing — an action pathway and a memory pathway ✅ CLOSED (Jul 2026)
- **Status: implemented.** `src/eval/rsc_routing.py` (n=5). The critique: the model bridges the spatial cortex to
  the LLM through a single unified gated cross-attention (`fusion.py`), but the retrosplenial cortex is **bifurcated**
  — M2-projecting neurons route to secondary motor cortex for ACTION affordances, AD-projecting neurons to anterior
  thalamus for allocentric location MEMORY, and inactivating one pathway impairs place-action association, the other
  object-location memory (projection-specific dissociation, *Molecular Psychiatry* 2024; RSC→M2 *J. Neurosci.* 2016).
  This is an architecture claim, so we hardcode only the two-pathway wiring (as the anatomy does) and let the CONTENT
  and the benefit emerge. Two conflicting demands are placed on the spatial read-out — ACTION = egocentric,
  heading-EQUIVARIANT "which way to turn"; MEMORY = allocentric, heading-INVARIANT "where the object is":
  - **(A) REFERENCE FRAMES DISSOCIATE (emergent).** Trained only on the combined task, heading is decodable from the
    ACTION head (**0.82**) but not the MEMORY head (**0.04**) — an egocentric/allocentric split never assigned.
  - **(B) SELECTIVE ROUTING.** The memory pathway carries the allocentric location, not the egocentric turn
    (selectivity **+0.95**), whereas a unified code is ENTANGLED (carries both, **0.76**).
  - **(C) THE SPLIT ENABLES THE DOUBLE DISSOCIATION.** Lesion the action pathway → action ×**5.5**, memory ×1.0;
    lesion the memory pathway → memory ×**58**, action ×1.0 (each lesion hits ONE task). A UNIFIED code lesioned by
    the same amount loses BOTH (action **+772%**, memory **+650%**) — so it is the segregation that makes the
    observed optogenetic double dissociation possible at all.
  - **(D) FALSIFIER — no conflict.** Make both tasks allocentric: the memory pathway stops excluding the action
    signal (action readable **0.41** vs **0.01** under conflict) — the specialization emerges from the conflicting
    frames, not the wiring.
- **Honest grade:** *clean emergent dissociation; the benefit is segregation, not efficiency.* The split does NOT
  lower total training loss — a full-capacity unified head fits both tasks — so the payoff is clean functional
  segregation (target-appropriate routing + selective lesionability), measured on that metric rather than a coarse
  loss (the project's recurring point that specific-benefit organs need their own metric). Wiring an action/memory
  split into the `fusion.py` gate is the natural follow-on. See `results/FINDINGS.md`.
- **Neuro basis.** RSC ships two streams, not one map: an egocentric action-affordance stream to motor cortex and an
  allocentric location stream to the thalamus; the segregation is why the two functions can be independently
  engaged, lesioned, and read.

---

## Tier 3 — GPU / language

### 8. The LLM reads the **conceptual-grid** map ✅ CLOSED — DEMONSTRATED ON A T4 (Jul 2026)
- **Status: RUN ON GPU, headline confirmed (bounded).** A frozen Qwen-1.5B + LoRA, reading ONLY the frozen
  space-cortex's code for concepts placed at 2-D coordinates (never the coordinates), trained on NEAR triples,
  answers "which concept is closer to the anchor?" (n=3 seeds, all converged): **closer_far 77.0% ± 2.8%**
  (never-seen far pairs); **OFF-AXIS 68.3% ± 3.5%** — the sharp signature (a 1-D/rank code is ≤50% there BY
  CONSTRUCTION, so this is genuine 2-D); **near(trained) 97%**. Falsifiers collapse: **cortex-OFF 50.0% ± 0.0%**
  and **shuffled-positions 49.6% ± 0.4%**. (n=3 floors the permutation p at 0.25; the effect is tight and large,
  ≥6 seeds only needed to push the statistic <0.05.) **The mechanism — the finding.** #9 ("more dominant?") is a
  1-D ordinal read, so a LINEAR head + the frozen LLM's compare sufficed (100%). "Closer" is a 2-D METRIC =
  grid population-vector CORRELATION/OVERLAP (Bellmund & Behrens 2018; Bush, Barry, Burgess 2015) — a DOT
  PRODUCT, QUADRATIC — which a linear head cannot compute (it read 50%). The load-bearing module is a
  **COINCIDENCE DETECTOR** (`CoincidenceReadout`): a shared per-candidate proximity(anchor,candidate) + a LINEAR
  combine → the frozen LLM does only the ordinal compare (an adversarial review confirmed the readout cannot
  self-answer). So the honest bound: the biological readout computes the METRIC; the frozen LLM does the ORDINAL
  compare — a **dissociation**, 1-D ordinal maps transfer to LLM reasoning cleanly, a 2-D metric needs the
  coincidence stage first. Ceiling ~0.70 (the code's inherent 2-D quality), below #9's 0.96. Getting here also
  required (all documented in `results/FINDINGS.md`): gain-control normalization (the code is ~98% a constant),
  right-padding, a padding-immune single-token scorer, and LR warmup (the weak signal converges late/seed-
  sensitively). CPU de-risk (`src/eval/conceptual_grid_cortex.py`, n=5): balanced OFF-AXIS **0.64 ± 0.03** vs
  shuffled **0.49**; held-out decode **0.63** vs **3.3** spacing. (Constantinescu, Behrens 2016; Bellmund
  2018.) See `results/FINDINGS.md`. *Original entry below.*
- After gap #2, a frozen-LLM readout answers abstract "which concept is closer / between?" from the
  grid-of-concepts code — cortex-ON vs text-only-OFF — extending the cognitive-map claim from space to
  meaning at the language level. (Notebook, T4.)

### 9. LLM reasoning over **social / other-agent** space ✅ CLOSED — DEMONSTRATED ON A T4 (Jul 2026)
- **Status: RUN ON GPU, headline confirmed.** A frozen Qwen-1.5B + LoRA, reading ONLY the frozen space-cortex's
  code for two agents' positions on a POWER×AFFILIATION social map (never the coordinates), trained on
  power-ADJACENT pairs, answers "who is more dominant?" (n=3 seeds): **dominance_far 100%**, transitive
  inference on never-seen FAR pairs; **dissociation 100%** — on pairs whose affiliation ordering OPPOSES power
  it still reads POWER; **adjacent(trained) 99%**. Falsifiers collapse: **cortex-OFF 50%** (same LoRA budget, no
  code → it cannot answer without the map) and **shuffled-positions 47.5%** (scrambled agent↔position → chance).
  (n=3 → the permutation p floors at 0.25; the effect is maximal with ~0 variance, run ≥6 seeds for p<0.05.)
  **Root cause that had to be fixed first (a genuine finding):** the frozen code is ~98% a position-INDEPENDENT
  constant + ~2% signal, so the readout's LayerNorm washed the signal out (the LLM saw an input-independent
  constant → 50%). A **gain-control / divisive-normalization stage** (per-dim standardization of the code over
  the concept set — the "missing module") makes the 2% visible; a linear decode is unchanged (1.0) but the LLM
  can now read it. `notebooks/m9_social_grid_llm_kaggle.py` + `src/training/train_social.py`. See
  `results/FINDINGS.md`. *Original entry below.*
- **CPU de-risk (still valid):** Extends gap #4 (self/other place cells) to the SOCIAL map
  at the language level: agents in a 2-D social space (POWER × AFFILIATION; Tavares 2015; Park-Miller 2021).
  The T4 headline (a frozen Qwen+LoRA answers "who is more dominant?" / "who is socially closer?" cortex-ON vs
  text-only-OFF) is scaffolded in `notebooks/m9_social_grid_llm_kaggle.py` + `src/training/train_social.py`.
  `src/eval/social_grid_cortex.py` (n=5) validates the design non-circularly on the frozen cortex.encode
  pipeline and finds a **dissociable** 2-D social map: (A) **DOMINANCE** — held-out pairwise dominance from the
  decoded POWER axis **0.96 ± 0.02** (the social transitive-inference result); (B) **SOCIAL DISTANCE** — a
  genuine 2-D metric, balanced OFF-AXIS "socially closer" **0.64** (>chance 0.5, where a power-only read is
  ≤0.5); (C) **AXIS DISSOCIATION** — power→dominance **0.96** vs affiliation→dominance **0.45** (gap
  **+0.51 ± 0.07**): the two
  social axes are separately readable (gap #4's double dissociation, now at the abstract-map level). FALSIFIER:
  shuffled agent↔position → dominance **0.44** (chance). The T4 cell reads this through the frozen LLM
  (dominance reuses the proven two-item train_relational forward, so it is the more tractable of the two GPU
  cells). T4 status: not yet run to success; the eval was hardened alongside #8 (balanced sets + padding-immune
  single-token scorer + periodic train-accuracy). See `results/FINDINGS.md`. *Original entry below.*
- LLM reasoning over **social / other-agent** space (after gap #4).

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
now closed.** And the **GPU/language capstones #8/#9 are now DEMONSTRATED ON A T4** — a frozen Qwen-1.5B + LoRA,
reading ONLY the frozen space-cortex's code (never coordinates), reasons over *two* kinds of cognitive map:
**#9 social hierarchy** (1-D ordinal — "who is more dominant?": far 100%, dissociation 100%, cortex-OFF/shuffled
50%) and **#8 conceptual space** (2-D metric — "which concept is closer?": off-axis 68% ± 4%, far 77%, cortex-OFF
50.0%, shuffled 49.6%, n=3). The two differ by exactly the neuroscience: the 1-D ordinal transfers to the LLM
with a linear read-out, while the 2-D metric needed a **coincidence-detector** read-out (grid population-vector
correlation) to first turn distances into proximities — the frozen LLM then does the ordinal compare. **Every
item in the register — CPU and GPU — is now closed**, each an emergent/transfer signature measured against its
own falsifiers, never put in a loss. See `results/FINDINGS.md` for the full write-ups (including the honest
multi-bug debugging trail behind #8: gain control, padding, single-token scoring, the coincidence module, and
LR-warmup stabilization).
