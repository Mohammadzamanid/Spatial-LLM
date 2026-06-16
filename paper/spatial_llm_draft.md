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
shuffling the metric (p=0.009).

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
magnitude. This is the representation-level validation of the headline LLM experiment (§7 roadmap), where
the readout is a frozen Qwen+LoRA answering a *linguistic* comparison it cannot do text-only.

## 7. Language transfer ✅ (n=3; ➕ more seeds to resolve grid-vs-place)

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
- §7 is n=3 with large seed variance; the grid-vs-place comparison there is inconclusive (needs n≥8).
  Emergence, boundary, replay pillars are demonstrations.

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
- ✅ §7 multi-seed LLM (n=3): cortex ON ≫ text-only OFF (robust); grid-vs-place inconclusive at n=3.
- ➕ optional: n≥8 LLM seeds to resolve the (modest, bearing-trending) grid-vs-place effect.
- ✎ tighten abstract/intro/related work; assemble figure panels; expand Methods/Extended Data.
- Framing locked: honest characterization (wins, ties, boundaries) + integrative demo; **no uniqueness
  claim**.
