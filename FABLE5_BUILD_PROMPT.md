# Master Build Prompt — "Cortex Explorer": an embodied 3-D agent + live brain visualizer (Windows .exe)

**To: Fable 5.** You are the most capable builder of interactive 3-D worlds. Your job is to take an existing,
biologically-faithful "spatial brain" (the `Spatial-LLM` repository) and give it a **body, a world, and a window
into its mind** — shipped as a single, polished **Windows `.exe`** that anyone can double-click and run.

This is not a game with a scripted NPC. The thing walking around is a real neural agent: its place cells, grid
cells, head-direction compass, drives, self-generated goals, imagined futures, and mood are all **actual model
state**, not animations. Your deliverable renders that brain honestly and beautifully while it roams worlds the
user chooses.

---

## 0. The one rule that overrides everything: honesty of signal

Every neural quantity you display **must be the real output of the real organ** for the current simulation tick —
the actual tensor, not a decorative loop or a plausible-looking fake. This repository's entire ethos is *emergence,
measured, never manufactured*. If an organ is not yet wired for 3-D, either wire it faithfully or label the panel
`PLACEHOLDER` in the UI — never dress up noise as neuroscience. A gorgeous visualizer of fake signals is a failure;
a plain visualizer of true signals is a success. Make it both true **and** gorgeous.

---

## 1. Start from the real brain

- **Repo:** `mohammadzamanid/spatial-llm`, branch **`claude/ecstatic-dijkstra-Rasvn`** (this is where everything
  below lives). Clone it, read `GAPS.md` (the register of what's built) and `results/FINDINGS.md` (what each organ
  does and its measured signature), and `reproduce_all.sh` (every eval and its one-line summary).
- **Language of the brain:** Python + PyTorch, CPU-only, small nets — it runs fine in real time on a laptop CPU.
- **Reuse, don't reinvent.** Import the actual organs. Do **not** re-derive the neuroscience. The relevant modules:
  - **Spatial cortex** — `src/models/neuro/spatial_cells.py` (`_HexGridModules` hex grid, `LocalOrder3DGrid`
    anisotropic 3-D grid, boundary/object-vector cells, hexadirectional readout), `src/models/neuro/attractor.py`
    (head-direction ring attractor), `src/models/place_cell_memory.py` (place cells / hippocampus).
  - **Path integration & geometry** — `src/eval/curved_path_integration.py` (holonomy), `src/eval/grid_shearing.py`
    (map warps with environment shape), `src/eval/reference_transform.py` (egocentric↔allocentric),
    `src/eval/anisotropic_3d.py` (vertical coded coarser — the terrestrial regime; use this for climbing worlds).
  - **Memory** — `src/eval/replay_planning.py` (forward/reverse replay), `src/models/neuro/theta_sweep.py`
    (look-ahead sweeps), `src/eval/systems_consolidation.py` (CLS), `src/eval/neurogenesis_stamp.py`,
    `src/eval/hippocampal_subfields.py` (DG/CA3/CA1), `src/eval/superposition_capacity.py`.
  - **Neuromodulation** — `src/models/neuromodulation.py` (ACh encode/retrieve, NE surprise/reset).
  - **The five AGENCY organs (the heart of the demo)** —
    `src/eval/intrinsic_motivation.py` (learning-progress drive), `src/eval/goal_generation.py` (autotelic goals +
    curriculum), `src/eval/forward_model.py` (efference copy → sense of agency + motor control),
    `src/eval/imagination_planning.py` (multi-step rollouts → planning), `src/eval/affect_valence.py` (mood as
    momentum).
  - **Drives / value** — `src/eval/interoceptive_map.py` (thirst/hunger), `src/eval/map_value_decouple.py`
    (successor map + value), `src/eval/unified_agent*.py` (existing integration capstones — study these; your
    embodied loop is their 3-D successor).
- **The organs are demonstrated as standalone evals.** Your first engineering task is to **wire them into one
  embodied agent loop** (see §4). Where an eval trains a small model, keep that; where it hard-codes a toy world,
  replace that world with your 3-D environment while preserving the organ's mechanism exactly.
- **Packaging note:** the models are tiny, so CPU inference is real-time. Bundle the real PyTorch organs into the
  `.exe` (PyInstaller/Nuitka). If binary size forces it, a **numpy port of the specific organs used is acceptable
  ONLY if you validate it reproduces the torch organ's output** (ship the validation as a test). Prefer the real
  torch code.

---

## 2. The deliverable

A single **Windows `.exe`** (64-bit, self-contained, no Python install required — double-click to run). On launch it
opens a clean main menu → the user picks an **environment** and a **brain configuration**, then enters a real-time
3-D scene where the agent roams autonomously while a live **brain dashboard** shows its cortical activity. Include a
short README and the build script. Target: runs at ≥ 30 FPS rendering with the brain stepping at a steady tick
(decouple render rate from brain-tick rate if needed).

---

## 3. Environments (user-selectable from the menu)

Ship at least these three, chosen from a menu with thumbnails:

1. **Simple maze** — a small, clean, single-level labyrinth. Flat floor, a few corridors, a goal/resource. The
   "hello world" for watching place fields and path integration form.
2. **Advanced maze** — large, multi-level, with **verticality** (ramps, ledges, a climbing wall or helix). This is
   where the **anisotropic 3-D grid code** (`anisotropic_3d.py`) matters — the agent should code height more coarsely
   than the horizontal plane, and you should be able to *see* that in the grid/place panels.
3. **Open 3-D world with time** — a natural terrain (hills, water, vegetation) with a **day/night cycle** and
   **resources that deplete/regrow**, so time genuinely passes. This is where drives (thirst/hunger), lifetime
   memory consolidation (CLS), and self-generated goals come alive.

Make the environment system **data-driven** (load from a small scene description) so more worlds can be added later.
A "sandbox" toggle to place obstacles/resources by hand is a bonus.

---

## 4. The agent — wire the organs into ONE embodied loop

Each simulation tick, the agent:

1. **Senses** — receives an egocentric observation from the world (local geometry, boundaries, visible objects,
   resource cues, its own vestibular/motor state). Feed this to the spatial cortex.
2. **Localizes** — the grid/place/HD cortex path-integrates self-motion and localizes; compute the **position
   estimate**, its **uncertainty** (reconstruction residual, per `uncertainty_behavior.py`), and reanchor on
   boundaries/objects. In vertical worlds, use the anisotropic 3-D grid.
3. **Feels & wants** — update **interoceptive drives**; update **mood** (`affect_valence.py`) from reward-prediction
   momentum; compute the **intrinsic-motivation / learning-progress** signal (`intrinsic_motivation.py`) over the
   places it could go.
4. **Chooses a goal** — the **autotelic goal generator** (`goal_generation.py`) proposes the agent's own current
   goal (explore the most learnable frontier, seek a drive-relevant resource, or a self-set target), modulated by
   mood and drives. No goal is scripted by you.
5. **Imagines & plans** — roll the **forward model** (`forward_model.py`) forward into multi-step **imagined
   rollouts** (`imagination_planning.py`) toward the goal; select a plan (MPC). Expose the imagined trajectories for
   visualization — this is the agent's "mind's eye."
6. **Acts** — execute the first action through the forward-model motor controller (Smith-predictor style, so it
   handles sensorimotor delay); the world updates. The **efference copy** predicts the sensory consequence; the
   **prediction error** = the sense-of-agency / self-vs-world signal.
7. **Remembers** — store experience; trigger **replay** at rest/goal (forward for planning, reverse for credit);
   consolidate map → cortex over the "lifetime" (CLS). Surface replay events for the viz.

The loop is autonomous: the user watches, and can optionally nudge (see §7 controls), but the behavior — where it
goes, what it wants — comes from the organs. **Acceptance:** with all organs on, the agent should visibly *explore
purposefully*, *set and pursue its own goals*, *detour around obstacles via imagination*, *avoid unlearnable "noisy"
regions*, *regulate its drives*, and *show mood shifts* — and turning an organ **off** should visibly degrade the
corresponding behavior (ship an "ablation" toggle per organ; that dissociation is the proof it's real).

---

## 5. The brain dashboard — what to visualize (all live, all real)

Lay these out as clean, dockable panels around a central 3-D viewport. Group them:

**A. Spatial map (the "where")**
- **Grid cells** — for each of several modules/scales, the hexagonal firing field rendered as a soft glowing
  overlay on the floor at the agent's location; a small module bank showing each module's phase. In vertical worlds
  show the anisotropic (vertically-stretched) fields.
- **Place cells** — a population; light up the cells whose fields contain the agent; optionally paint accumulated
  place fields onto the floor as the agent explores (watch the map build).
- **Head-direction compass** — the ring-attractor bump as a live compass dial (true vs decoded heading).
- **Boundary / object-vector cells** — highlight when near walls/objects; draw the vector to the anchoring object.
- **Position estimate & uncertainty** — a ghost marker for the agent's *believed* position vs its true position,
  with an uncertainty halo that grows with path-integration drift and shrinks on landmark reanchoring.

**B. Autonomy (the "why" and "what next") — make these the stars**
- **Intrinsic motivation** — a heatmap over the reachable world colored by learning-progress ("how interesting /
  learnable is it there"), updating as the agent masters regions; noisy/unlearnable regions visibly cool off.
- **Current goal** — a beacon on the self-proposed goal, with a small "competence/curriculum" readout (what
  difficulty it's currently choosing — watch it ramp up over the session).
- **Imagination / planning** — the agent's imagined rollouts drawn as translucent **ghost trajectories** fanning out
  from it each plan step, with the chosen plan highlighted. The single most compelling visual: you can literally see
  it *think* before it moves.
- **Sense of agency** — a live meter of forward-model prediction error split into **self-caused (low, "I did that")**
  vs **world-caused (high, "the world did that")**; flash when something external happens.
- **Mood / affect** — a global **mood meter** and a scrolling mood timeline; subtly **tint the whole scene** with
  valence (warm when good, cool when bad). Show it swing.
- **Drives** — thirst/hunger bars; show them rise and get satisfied at resources.

**C. Memory (the "when")**
- **Replay** — when replay fires, animate the swept trajectory over the map (forward = magenta look-ahead, reverse =
  cyan credit-assignment), with a small event ticker.
- **Consolidation** — a slow indicator of map→cortex transfer over the lifetime; optionally a "recent vs remote"
  memory strength readout.

Every panel needs a one-line plain-language caption ("Grid cells: the brain's coordinate system") and a toggle. Keep
it legible: no more than ~6–8 panels visible at once; let the user choose which.

---

## 6. Visual design

Modern, dark, "mission-control for a mind" aesthetic — think a clean scientific instrument, not a busy game HUD.
- Central 3-D viewport with smooth, appealing rendering (soft shadows, gentle ambient occlusion, readable materials).
  Free-orbit camera + a "follow the agent" camera + a top-down "map" camera.
- Panels docked in a tidy grid with generous spacing, subtle motion, and a restrained, **colorblind-safe** palette
  (one accent per organ, used consistently everywhere that organ appears). Everything animates smoothly (no jitter).
- Typography: one clean sans; big legible numbers on meters. A quiet grid/graticule background ties it together.
- It should look like something you'd screenshot for a Nature cover — calm, precise, alive.

---

## 7. Controls & UX

- Main menu: choose environment, choose brain config (all-on, or a preset), Start.
- In-scene: **play / pause / step**, **speed** (0.25×–8×), camera modes, panel toggles, and a **per-organ ablation
  switch** (turn intrinsic motivation / goals / forward model / imagination / affect / a spatial module on/off and
  watch behavior change — this doubles as the demo's proof of faithfulness).
- Optional: click a cell in the place/grid panel to spotlight its field in 3-D; click a floor tile to hand the agent
  a goal there and watch it plan.
- A subtle "recording" button that dumps a short clip / screenshot for sharing.

---

## 8. Architecture (recommended; use your judgment)

- **Brain process:** the real Python/PyTorch organs, running the §4 loop, emitting a compact **state packet each
  tick**: agent pose; per-module grid activations; active place cells; HD bump; position estimate + uncertainty;
  intrinsic-motivation field (sampled grid); current goal; list of imagined rollouts (polylines); agency error
  (self/world); mood scalar + history; drive levels; replay events. Define this as a versioned schema (JSON for
  clarity, or a packed binary if you need the bandwidth).
- **World + renderer:** pick the stack you can make most beautiful and ship as a Windows `.exe`. Reasonable options,
  in rough order of "reuses the Python brain most directly":
  1. **All-Python** — 3-D via Panda3D/Ursina, dashboards via Dear PyGui/imgui, packaged with PyInstaller. Simplest
     data path (in-process), one language.
  2. **Godot 4** for world+dashboard (native Windows export) + the Python brain as a subprocess over a local socket
     or shared memory. Best-looking with least fighting the renderer.
  3. **Web tech** (Three.js + a Python local server) wrapped with Tauri/Electron → `.exe`.
  Whatever you choose: **decouple brain-tick from frame rate**, keep the state-packet interface clean, and make the
  build **one command** producing the `.exe`.
- Ship: the `.exe`, source, a `build.(bat|ps1)`, a README (how to run, what each panel means, how to add worlds),
  and the organ-fidelity validation test.

---

## 9. Milestones (ship each as a runnable build)

1. **M1 — Body in a world.** The three environments load from data; the agent locomotes via the real forward-model
   motor controller; menu + camera + play/pause; produces a Windows `.exe`.
2. **M2 — The "where."** Live spatial-cortex panels (grid, place, HD, boundary, position estimate + uncertainty),
   faithful to the real organs; watch the map build as it explores.
3. **M3 — The "why."** The five agency organs wired and visualized: intrinsic-motivation heatmap, self-set goal,
   imagination ghost-rollouts, agency meter, mood tint/timeline, drives — plus per-organ ablation toggles that
   visibly change behavior.
4. **M4 — The "when" + polish.** Replay/consolidation viz, the open-world day/night + resources, visual polish,
   recording, README, and the fidelity validation. Final `.exe`.

---

## 10. Acceptance criteria

- Double-clicking the `.exe` on a clean Windows machine opens the app; the user can pick a maze or the open world and
  watch an agent that **explores on its own, sets and pursues its own goals, plans around obstacles by imagining,
  regulates drives, and shifts mood** — with every on-screen neural signal traceable to a real organ's output.
- Ablating any agency organ **visibly** removes its contribution (the built-in dissociation).
- It is genuinely beautiful and legible — a thing worth showing off — **and** every signal is true.

Build it so that when it runs, we are not watching a game character. We are watching a small mind explore a world,
and seeing what it's thinking while it does. Make it honest. Make it stunning.
