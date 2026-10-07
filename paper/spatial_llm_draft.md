# Causal cognitive-map interfaces for language models: when brain-inspired spatial codes help, and when they do not

**Submission draft — Spatial-LLM.** Every quantitative claim below is tied to a committed script or notebook and raw result artifact. We distinguish properties that **emerge under learning** from properties imposed as **architectural priors**, and report null/negative results as first-class findings.

---

## Abstract

Language models can describe space without maintaining a metric state that survives self-motion. We test whether a structured cognitive-map interface supplies that missing state. Self-motion is integrated into a bounded multi-scale population code and injected into a frozen language model through gated cross-attention. Causal ablations show that spatial answers depend on this latent channel and on task-relevant subcodes. A prospective theta-sweep signal improves blocked-ahead reasoning in Qwen2.5-1.5B and replicates in SmolLM2-1.7B (ON 69.9% versus NO-SWEEP 49.8%, p=0.0156). Ordinary Euclidean path integration is matched by a simpler additive Transformer, establishing an important boundary condition. On user-disjoint Microsoft GeoLife trajectories, the fixed representation reaches 76.6% bearing and 92.3% distance accuracy, far above OFF baselines. Structured cognitive maps therefore provide causal spatial state to language models, with benefits determined by task-specific representational demands.

---

Coordinate embeddings let a model memorize a map; they do not obviously let it *compute* over space in
a way that survives a change of scale or serves many downstream uses. The mammalian
entorhinal–hippocampal system solves navigation with a particular representational scheme — grid cells
that path-integrate velocity into a periodic multi-scale code, read out by place cells — that also
appears to underlie planning, value, and relational cognition. We ask a direct question: if we build
that substrate, self-supervised and label-free, and let a frozen language model read it, **what does it
buy, and what does it not?** We answer with fair baselines and multiple seeds throughout, and we let the
negative results stand.

## Results

### System and causal interface

A path of self-motion → conjunctive velocity cells → a **biologically constrained** velocity-driven hexagonal grid code (fixed gains, geometric scale ratios; phase = gain·∫v wrapped on a hexagonal torus) → a learned place/value readout → gated cross-attention into a frozen Qwen2.5-1.5B + LoRA. The cortex is pre-trained only to
predict bounded place-cell activity from self-motion (no coordinate labels). Architecture and configs:
`src/models/`, `results/architecture.svg`.
**Claim calibration.** The hexagonal geometry of the publication model is **not** claimed to emerge from an unconstrained network. The unconstrained attractor develops periodic multi-field responses but not hexagonal symmetry (mean gridness −0.46). The hexagonal publication condition uses an explicit biologically motivated toroidal lattice / velocity-driven module and yields mean gridness +0.87. We therefore treat hexagonality as an inductive bias and ask what causal computational role the resulting bounded periodic code plays downstream.


### Length generalization and bounded representations

Stripping away the LLM (so any effect is the representation), an agent random-walks in 2-D; we train a
position readout on mixed short paths {6,8,10,12} (scale-free) and test to 4× longer, deriving the
trajectory-QA tasks from the decoded displacement (`src/eval/extrapolation.py`, n=8). Against a *fair*
place baseline (tiled exactly to the trained region), the grid code wins at every length — at 3×,
**93% ±0 vs 80% ±1** distance accuracy, non-overlapping CIs — because a bounded place code cliffs once
paths leave its trained box while the grid code degrades gracefully (its phase is scale-free *and*
periodic). An exact-integration oracle is flat, so the gap is the code, not the task. (Fig. 2; `results/extrapolation.json`. Honest ceiling: grid itself falls to 75% at 4×; range is finite.)

### Additive integration explains ordinary Euclidean path integration

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
grid cells. (Fig. 2; `results/ablations.json`, `results/seq_baselines.json`.)

### Regime-dependent value of structured population codes

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
extrapolation, labelled multi-map, noise), and loses only in the very-low-data regime. (Figs. 2–3; `results/code_necessity.json`, `results/multimap_task.json`, `results/frontier_probes.json`, `results/phase_diagram.json`.)

**Significance (paired tests; Fig. 2; `src/eval/significance.py`).** Every claimed effect is
statistically significant under a paired sign-flip permutation test with a bootstrap CI of the
difference (n=20 fast / n=8 heavy): grid−place distance@T24 Δ=+0.124, p<1e-4, d=10.9 (20/20 seeds);
grid+remap−additive multi-map Δ=+0.766, p<1e-4; population−raw-2D capacity Δ=+0.507, p<1e-4;
Hebbian−gradient Δ=+0.662, p<1e-4; value−random goal-nav Δ=+0.670, p=0.006; transitive-inference−chance
Δ=+0.338, p<1e-4. Critically, the **honest null is certified**, not assumed: grid vs a NoPE+sum
Transformer on path integration is Δ=+0.002, 95% CI [−0.022,+0.032], **p=0.94, d=0.04**.
(`results/significance.svg`.)

**The tie inverts on non-Euclidean worlds — where the periodic code is *necessary* (Fig. 3; `src/eval/torus.py`).** On a torus, true position is θ = (∫velocity) mod 2π; a periodic grid code
computes that mod for free (cos ∫v = cos θ at any wrap count) while a non-periodic code sees an unbounded
∫v and cannot recover the wrap. Trained on short paths and tested to many wraps (n=8), the grid code is
**flat at the oracle floor (0.01 rad, 100% within 45°) at every length**, while the *same NoPE+sum
Transformer that tied it on Euclidean paths collapses to chance (1.56 rad, 25%)**, as do additive and
Euclidean-place codes — tiny, non-overlapping CIs. So the periodicity that was a wash on Euclidean paths
is exactly the right inductive bias for a cyclic world: there the brain-faithful code is not
competitive-but-tied, it is **necessary**. This is also the **leakage rebuttal** — a torus has no
faithful Euclidean text description, so a language prior cannot substitute for having path-integrated it.

### Causal language transfer through the spatial channel

The main causal transfer results are summarized in **Fig. 1**.

We next ask whether a language model actually uses the latent map rather than solving the task from text. In all headline language experiments, the move sequence is withheld from the prompt and reaches the model only through the spatial channel; cortex-OFF therefore provides a direct leakage control.

**Single-item spatial readouts.** A frozen Qwen2.5-1.5B + LoRA reads path-integrated state from the cortex well above text-only OFF. On the non-Euclidean torus task, cortex-ON reaches **84/74/63%** at T=8/16/24 versus **~9–11%** OFF, with ON>OFF in all six seeds and paired sign-flip **p=0.033** at every length (`results/torus_llm.json`). Because the task depends on wrap-around state that is never described in text, this is the cleanest demonstration that the answer is carried by the integrated spatial representation rather than a language prior. The temporal analogue behaves similarly: elapsed-time EXACT accuracy is **55% ±20** versus **16% ±6** OFF and WITHIN-1 is **70% ±19** versus **37% ±17**, again ON>OFF in all six seeds (**p=0.033**; `results/elapsed_time_llm.json`).

**Organ-specific causal dissociation.** The language readout is not merely sensitive to “some extra vector.” In the unified dead-reckoning experiment, WHERE depends on the grid/position organ and FACING on the head-direction organ: ablating the relevant organ selectively collapses its own readout while sparing the other (`results/deadreckoning_llm_agg.json`). Likewise, in the multi-reference-frame experiment, allocentric WHERE collapses under grid ablation but survives object-vector ablation, whereas egocentric LANDMARK collapses under object-vector ablation but survives grid ablation (`results/multiframe_llm_agg.json`). These double dissociations establish causal specificity of the latent subcodes.

**Boundary condition.** A separate n=3 grid-vs-place LLM comparison is intentionally not used as a headline superiority claim: cortex-ON is far above OFF, but grid vs place is not statistically separable at that sample size (`results/extrapolation_llm.json`). This agrees with the representation-level characterization in Sections 3–5: the robust language claim is the causal usefulness of a structured spatial state, not universal grid dominance.

### Prospective theta-sweep and cross-backbone replication

Prospective ablations and the second-backbone replication are summarized in **Fig. 4**.

A cognitive map is useful not only for representing current state but also for sampling what lies ahead. We therefore expose prospective theta-sweep tokens generated from the grid map and ask a blocked-ahead question in novel per-episode layouts, where the answer cannot be inferred from current position alone.

At the representation/readout level, the real sweep achieves **0.90** accuracy versus **0.58** with the sweep ablated and **0.63** with a wrong-heading sweep (`results/theta_sweep_readout.json`). The same causal pattern transfers to a frozen Qwen language model: across **8 seeds**, cortex-ON beats OFF, NO-SWEEP, and wrong-heading controls in every seed (**paired p=0.0081**; `results/theta_sweep_llm_agg.json`). This is the strongest prospective-language experiment because the ablation removes exactly the information required to answer.

We then repeated the experiment on a second open-weight family, **SmolLM2-1.7B-Instruct**, using a forced-choice 0/1 evaluation to remove generation-format confounds. Across **8 seeds**, ON reaches **69.9% ±12.6** versus **49.8% ±0.3** NO-SWEEP, a **+20.1 percentage-point** effect with **7 wins, 0 losses, 1 tie** and exact paired sign-flip **p=0.0156**. ON also exceeds text-only OFF by **+19.9 points** and wrong-heading sweep by **+14.4 points** (both **p=0.0156**; `results/sweep_llm_smollm2_v2.json`). Absolute accuracy remains seed-variable, so the replication claim is directional and causal rather than an assertion of identical convergence across backbones.

Together, the Qwen and SmolLM2 results rule out the simplest backbone-specific explanation: prospective information supplied by the cognitive-map channel changes language-model behavior in the predicted direction across two materially different open-weight LLM families.

### Scope of the main claim

The repository contains additional biologically inspired components—successor representations, time-cell analyses, replay, semantic warping, boundary/object reanchoring, 3-D/local-order grid variants, content binding, basal-ganglia action selection, and closed-loop behaving agents. These analyses are retained in `paper/SUPPLEMENTARY_RESULTS.md` and the corresponding committed result artifacts.

They are intentionally not part of the flagship causal spine. The main paper requires only four claims:

1. a latent cognitive-map channel causally supplies spatial state to a frozen LLM;
2. ordinary Euclidean integration can be matched by simpler additive mechanisms, while periodicity/remapping/population structure matter in specific regimes;
3. the strongest prospective theta-sweep effect replicates across Qwen and SmolLM2; and
4. the fixed representation retains useful metric information on real, user-disjoint GeoLife trajectories.

This separation prevents auxiliary neurobiological demonstrations from being mistaken for necessary premises of the central result.

### External validation on real human trajectories

The preregistered external-validation results are summarized in **Fig. 5**.

The controlled experiments above use synthetic/self-generated motion so that spatial-channel interventions can be isolated exactly. We therefore tested the fixed spatial code on an external real-world dataset, **Microsoft GeoLife GPS Trajectories 1.3**, without changing the grid architecture. GPS traces were converted to east/north self-motion and divided by user into seeded-random **70/15/15 user-disjoint** train/validation/test splits. Spatial scaling and distance-bin cut-points were estimated from TRAIN users only. We evaluated non-overlapping trajectory windows at T={8,16,24}; the inferential unit is the held-out **user**, not the readout seed.

The preregistered primary task was 8-way endpoint bearing. Across eight readout seeds, the fixed grid population reached **76.6%** accuracy versus a train-majority OFF baseline of **16.2%**. Across **28 held-out users**, the paired GRID−OFF effect was **+59.3 percentage points**, bootstrap 95% CI **[+50.5,+67.5]**, with a user-level sign-flip **p≈1×10⁻⁵**. The secondary 6-bin endpoint-distance task reached **92.3%** versus **18.4%** OFF; the held-out-user effect was **+76.5 points**, 95% CI **[+72.5,+80.3]**, **p≈1×10⁻⁵**. (`results/geolife_external_v2.json`.)

A corrected exact-displacement calibration provides the appropriate ceiling: a small RAW-MLP given the exact additive endpoint vector reaches **98.7%** bearing and **98.8%** distance (analytic RAW oracle = 100%). The bounded grid code is therefore **not superior to explicit Cartesian integration**, nor should it be; the external result shows that a fixed bounded periodic neural population retains enough metric information to support accurate decoding on real trajectories from unseen people. This closes the external-validity gap without changing the paper's regime-dependent claim.

## Discussion

**Neural spatial codes.** Grid cells and path integration motivate the bounded periodic code (Hafting
2005; Burak & Fiete 2009), while trained recurrent integrators show that grid-like representations can
arise under navigation objectives (Banino 2018; Cueva & Wei 2018). Modular coding work explains the
range/capacity trade-off (Stensola 2012; Sreenivasan & Fiete 2011). We use these results as computational
priors rather than claiming that every anatomical detail is reproduced.

**Cognitive maps beyond physical space.** The Tolman–Eichenbaum Machine and concept-space results motivate
relational transfer (Whittington 2020; Constantinescu 2016), while Complementary Learning Systems motivates
the separation between rapid episodic storage and slower parametric learning (McClelland, McNaughton &
O'Reilly 1995).

**Spatial reasoning in foundation models.** Recent benchmarks increasingly test whether multimodal models
construct internal spatial models rather than merely recognize visible relations. VSI-Bench evaluates
configurational, metric, and spatiotemporal reasoning from egocentric videos, while MindCube (ICLR 2026)
tests cognitive mapping, perspective taking, and mental simulation from limited views and finds large gains
from an explicit map-then-reason scaffold. These benchmarks are complementary rather than directly
interchangeable with our experiments: they begin from visual observations, whereas our causal tests isolate
the effect of a latent self-motion/cognitive-map channel on a language model. Extending the present cortex
with a visual scene encoder and evaluating on those benchmarks is therefore a separate multimodal question,
not a drop-in validation of the current system.

Our contribution is a controlled causal characterization: we intervene on the spatial channel and its
subcodes, include fair non-neural baselines and certified nulls, and ask **when** a structured cognitive map
changes a language model's behavior rather than assuming that brain-inspired structure is always beneficial.

The results also define clear limits. The central causal interventions use controlled synthetic environments, while representation-level external validation uses real GeoLife trajectories. The strongest theta-sweep effect replicates across Qwen2.5-1.5B and SmolLM2-1.7B, but the work does not establish natural multiview or video-based embodied reasoning. A NoPE+sum Transformer matches the grid code on ordinary Euclidean path integration, so grid coding is not the best or uniquely necessary pure integrator. Remapping and population-capacity advantages are regime-specific, particularly in fixed-memory and context-free settings. Finally, the n=3 grid-versus-place LLM comparison remains underpowered and is treated as inconclusive rather than a positive headline.

The main conclusion is therefore narrower than a general claim for brain-inspired superiority: a structured latent map can causally supply spatial state to a language model, while the usefulness of specific neural-style coding properties depends on the computational structure of the task.

## Methods

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

### AI-assisted development and writing

Generative AI tools (OpenAI ChatGPT) were used during development for code drafting, debugging support, literature-search assistance and editorial restructuring of the manuscript. All experimental designs, scientific claims, code changes, statistical interpretations and manuscript text were reviewed and accepted by the human author, who retains full responsibility for the work. Generative AI was not treated as an author and did not independently generate or alter experimental observations.

### Code and data availability

All scripts underlying the main claims, committed result JSONs, GPU notebooks, and reproducibility instructions are available in the Spatial-LLM repository. The exact submission artifact will be archived on Zenodo and the persistent DOI inserted here before submission. The Microsoft GeoLife source archive is not redistributed; the repository contains deterministic preprocessing code and the locked user-disjoint evaluation protocol needed to regenerate the external-validation benchmark from the official dataset.



### Status / path to submission
- ✅ Core CPU characterization and causal language readouts are committed with multi-seed artifacts.
- ✅ Claim language now distinguishes learned/emergent phenomena from the explicitly constrained hexagonal prior.
- ✅ Null results and non-convergent seeds remain reported rather than removed post hoc.
- ✅ Reproducibility instructions, raw JSON results, tests, and figures are in-repository.
- ✅ Second-backbone replication complete: SmolLM2-1.7B reproduces the theta-sweep causal effect (primary ON vs NO-SWEEP p=0.0156).
- ✅ External real-trajectory validation complete: GeoLife bearing and distance both exceed OFF across held-out users with p≈1×10⁻⁵; corrected exact-displacement baselines are reported as ceilings.
- ⚠️ The grid-vs-place LLM comparison remains underpowered at n=3 and is treated as inconclusive, not as a positive headline.
- Framing locked: causal cognitive-map interface + regime map of wins/ties/failures; **no claim that grid cells are universally superior or that hexagonal geometry emerged unconstrained**.
