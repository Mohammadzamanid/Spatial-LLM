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
suffices, together with the integrative demonstration. We do not claim grid cells are a uniquely
necessary substrate for a trained system, and we show why.

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
contribution. (Figures 3–4: `results/code_necessity.svg`, `results/multimap_task.svg`,
`results/frontier_probes.svg`.)

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

## 7. Language transfer ⏳

A LoRA-Qwen2.5-1.5B answers navigation questions through the frozen cortex (the moves reach the model
only via the cortex). Single-seed, the grid cortex beats the place/default cortex and stays flat to 3×
training length, with cortex-OFF at chance:

| task (cortex ON, T=8/16/24) | grid cortex | place/default |
|---|---|---|
| return | 100/100/100 | 96/89/86 |
| bearing | 85/83/80 | 71/78/73 |
| distance (exact) | 95/88/85 | 62/46/40 |

⏳ The multi-seed version (grid vs place × {distance,bearing} × seeds, 95% CI, + OFF control) is
specified and resumable on a single T4 (`notebooks/m2_extrapolation_multiseed_kaggle.py`) — the one
remaining GPU run, and the cleanest novel result (a frozen LLM reasoning through a self-supervised grid
code).

## 8. Related work ✎

Grid cells / path integration (Hafting 2005; Burak & Fiete 2009); grid codes in trained integrators
(Banino 2018; Cueva & Wei 2018); modular coding for range/capacity (Fiete; Stensola 2012; Sreenivasan &
Fiete 2011); the Tolman-Eichenbaum Machine and grid codes in concept space (Whittington 2020;
Constantinescu 2016); Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995); length
generalization in sequence models (the default does not generalize — the motivation for positional-
encoding research). Our contribution is the *fair, multi-seed characterization* of which of these
properties transfer to a trained model + the integrative LLM demonstration.

## 9. Limitations (honest) ✎

- The representation tasks are 2-D, unbiased random walk (~√T magnitude growth), single-T4 LLM scale.
- The headline "grid extrapolates" claim is matched by a NoPE+sum Transformer; the grid code is not the
  best pure path-integrator.
- The remapping/capacity advantages are regime-specific (fixed memory / context-free) and do not
  transfer to a trained LLM with a text context label.
- §7 is single-seed pending the Kaggle run; emergence, boundary, replay pillars are demonstrations.

## 10. Methods ✎

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
- ⏳ §7 multi-seed LLM table (one Kaggle run).
- ✎ tighten abstract/intro/related work; assemble figure panels; expand Methods/Extended Data.
- Framing locked: honest characterization (wins, ties, boundaries) + integrative demo; **no uniqueness
  claim**.
