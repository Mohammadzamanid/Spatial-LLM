# Master Build Prompt — "Agent Explorer": an autonomous AI agent in a 3-D world + live model-internals visualizer (Windows .exe)

**To: Fable 5.** You are the most capable builder of interactive 3-D applications. Your job: take an existing
modular **reinforcement-learning / model-based AI agent** (the `Spatial-LLM` codebase) and build a polished,
double-click **Windows `.exe`** that lets anyone watch this agent act autonomously in 3-D worlds they choose — while
a live dashboard renders the agent's **internal activations and state** as it runs. It is an interactive demo and a
model-introspection / debugging tool in one.

The character moving around is not scripted. It is driven by real trained modules — its learned spatial embeddings,
world model, planner, exploration drive, auto-generated goals, and a global reward-trend variable are all **actual
model tensors**, not hand-authored animation. Your deliverable renders that real computation honestly and
beautifully.

---

## 0. The one rule that overrides everything: fidelity of signal

Every number and visual you display **must be the real output of the real module** for the current simulation step —
the actual tensor, not a decorative loop, a random generator, or a plausible-looking placeholder. This is a debugging
and demonstration tool for a real system; a gorgeous dashboard of fake numbers is worthless and defeats the point. If
a module isn't wired into the 3-D loop yet, either wire it faithfully or mark that panel `PLACEHOLDER` in the UI —
never dress up noise as a live signal. Make it both **true** and **stunning**.

---

## 1. Start from the real agent (reuse, don't reimplement)

- **Codebase:** `mohammadzamanid/spatial-llm`, branch **`claude/ecstatic-dijkstra-Rasvn`**. Clone it. Read
  `GAPS.md` and `results/FINDINGS.md` for what each module does and its measured behavior, and `reproduce_all.sh`
  for a one-line summary of every module and how to run it.
- **Stack of the agent:** Python + PyTorch, CPU-only, small networks — it runs comfortably in real time on a laptop
  CPU. Import the existing classes; **do not re-derive the ML** — wire the existing modules into one agent loop.
- **The modules you'll drive** (find and import them under `src/models/` and `src/eval/`; several are currently
  written as standalone evaluation scripts you'll refactor into one live loop). By functional role:

  | Role (AI/ML function) | Where |
  |---|---|
  | **Spatial representation** — a multi-scale, periodic position encoder ("grid-style" code) + location-selective embedding units + a recurrent heading estimator + obstacle/landmark feature detectors | `src/models/` (the spatial-representation package) |
  | **Localization / dead-reckoning + belief** — integrate self-motion to an estimated pose, with an **uncertainty** estimate that grows with drift and shrinks when it re-anchors on landmarks | `src/eval/uncertainty_behavior.py`, the localization modules |
  | **3-D representation, anisotropic** — in vertical worlds the encoder learns to represent height at **lower resolution** than the horizontal plane (an emergent property of the training data, worth showing) | `src/eval/anisotropic_3d.py` |
  | **World model (dynamics model)** — predicts the next observation from the current state + the executed action; also yields a **self-caused vs externally-caused** change signal (its prediction error) and delay-compensated control | `src/eval/forward_model.py` |
  | **Model-based planning** — roll the world model forward over candidate action sequences and select (MPC); the rollouts are the agent's look-ahead | `src/eval/imagination_planning.py` |
  | **Intrinsic-motivation exploration** — a curiosity / learning-progress signal that drives exploration with no external reward, and avoids unlearnable ("noisy") regions | `src/eval/intrinsic_motivation.py` |
  | **Automatic goal generation (auto-curriculum)** — the agent proposes its own goals at the frontier of its competence; goal difficulty ramps up over a session | `src/eval/goal_generation.py` |
  | **Global reward-momentum state** — a slow scalar integrating whether recent outcomes beat expectations, which feeds back to bias perceived reward (a control loop; see §4) | the reward-momentum module under `src/eval/` |
  | **Internal resource variables** — energy/water-style levels the agent must keep regulated by reaching resources | the resource-regulation module under `src/eval/` |
  | **Experience replay + slow-weight consolidation** — replay stored trajectories (forward for planning, reverse for credit assignment); move fast episodic memory into slow network weights over a "lifetime" | `src/eval/replay_planning.py`, `src/eval/systems_consolidation.py` |
  | **Value / successor map** and the **integration reference loops** — how the existing capstones combine the modules | `src/eval/map_value_decouple.py`, `src/eval/unified_agent*.py` |
  | **Adaptive gates** — learning-rate / attention gating signals that modulate the pipeline | the gating modules under `src/models/` |

- **Packaging note:** the networks are tiny, so CPU inference is real-time; bundle the real PyTorch modules into the
  `.exe` (PyInstaller / Nuitka). If binary size forces it, a NumPy port of the specific modules used is acceptable
  **only if you ship a test proving it reproduces the PyTorch module's output** to tolerance. Prefer the real code.

---

## 2. The deliverable

A single **Windows `.exe`** (64-bit, self-contained, no Python install required — double-click to run). On launch it
opens a clean menu → the user picks an **environment** and an **agent configuration**, then enters a real-time 3-D
scene where the agent acts autonomously while a live **model-internals dashboard** shows its activations and state.
Ship the `.exe`, the source, a one-command build script, a README, and the fidelity test. Target ≥ 30 FPS render with
the agent stepping at a steady tick (decouple render rate from tick rate).

---

## 3. Environments (user-selectable from the menu, with thumbnails)

Ship at least three; make the environment system **data-driven** (loaded from a small scene description) so more can
be added later:

1. **Simple maze** — a small, clean, single-level labyrinth: flat floor, a few corridors, a goal/resource. The
   baseline for watching the spatial embedding and localization form.
2. **Advanced maze** — large, **multi-level**, with verticality (ramps, ledges, a climbing wall). This is where the
   **anisotropic 3-D encoder** matters — height is represented more coarsely than the horizontal plane, and you
   should be able to *see* that in the spatial panels.
3. **Open 3-D world with time** — natural terrain (hills, water, vegetation), a **day/night cycle**, and **resources
   that deplete and regrow**, so time genuinely passes. This is where the resource variables, lifetime consolidation,
   and self-generated goals come alive.

A "sandbox" toggle to place obstacles/resources by hand is a bonus.

---

## 4. The agent — wire the modules into ONE loop

Each simulation tick, the agent:

1. **Observes** — gets an egocentric observation from the world (local geometry, obstacles, visible objects/resource
   cues, its own motion/heading state).
2. **Encodes & localizes** — the spatial-representation modules integrate self-motion and produce a **pose estimate**,
   an **uncertainty** estimate, and re-anchor on landmarks/boundaries. In vertical worlds, use the anisotropic 3-D
   encoder.
3. **Updates internal state** — update **resource variables**; update the **global reward-momentum scalar** from
   reward-prediction momentum; compute the **curiosity / learning-progress** signal over reachable regions.
4. **Generates a goal** — the **auto-curriculum** module proposes the agent's own current goal (explore the most
   learnable frontier, seek a resource that a low resource-variable makes relevant, or a self-set target), modulated
   by the reward-momentum scalar and resource state. **No goal is scripted by you.**
5. **Plans by rolling out the world model** — roll the world model forward into multi-step candidate trajectories
   toward the goal and select one (MPC). Expose those rollouts for visualization — the agent's look-ahead.
6. **Acts** — execute the first action through the model-based (delay-compensated) controller; the world updates. The
   world model predicts the observation the action should cause; the **prediction error** flags **self-caused vs
   externally-caused** change.
7. **Stores, replays, consolidates** — store experience; trigger **replay** at rest/goal; slowly move episodic memory
   into network weights over the lifetime. Surface replay events.

The loop is autonomous — the user watches (and may optionally nudge; see §7), but *where it goes and what it wants*
come from the modules. **Acceptance:** with everything on, the agent should visibly **explore purposefully, set and
pursue its own goals, plan around obstacles via world-model rollouts, avoid unlearnable regions, keep its resource
variables regulated, and show reward-momentum shifts** — and toggling a module **off** should visibly degrade the
matching behavior (ship a per-module **ablation** switch; that dissociation is the built-in proof the signals are
real).

---

## 5. The dashboard — what to visualize (all live, all real tensors)

Clean, dockable panels around a central 3-D viewport, grouped:

**A. Spatial representation (the "where")**
- **Spatial code modules** — for several scales, the module's periodic activation as a soft glowing heatmap on the
  floor at the agent's location; a small bank showing each module's phase. In vertical worlds, show the anisotropic
  (vertically-stretched) activation.
- **Localization units** — a population of location-selective units; light up those active at the agent's spot;
  optionally paint their accumulated fields onto the floor as it explores (watch the representation build).
- **Heading estimate** — the recurrent orientation module's output as a live compass dial (true vs estimated).
- **Obstacle / landmark detectors** — highlight when near walls/objects; draw the vector to the anchoring object.
- **Localization belief & uncertainty** — a ghost marker for the *estimated* pose vs the true pose, with an
  uncertainty halo that grows with drift and shrinks on landmark re-anchoring.

**B. Autonomy (the "why" / "what next") — make these the stars**
- **Curiosity / learning-progress map** — a heatmap over the reachable world colored by learning progress ("how
  learnable is it there"), updating as the agent masters regions; noisy/unlearnable regions visibly cool off.
- **Auto-generated goal** — a beacon on the self-proposed goal, plus a small readout of the current difficulty the
  curriculum is choosing (watch it ramp over the session).
- **Model-based rollouts (planning)** — the candidate world-model trajectories drawn as translucent **ghost paths**
  fanning out each plan step, with the selected plan highlighted. The single most compelling visual: you can see it
  *look ahead* before it moves.
- **Self- vs externally-caused change** — a live meter of world-model prediction error split into **self-caused
  (low, the agent's own action)** vs **externally-caused (high, the environment)**; flash on external events.
- **Global reward-momentum state** — a big meter + a scrolling timeline; subtly **tint the whole scene** by its sign
  (warm when outcomes beat expectations, cool when they fall short). Show it swing.
- **Resource variables** — energy/water bars; show them fall and get replenished at resources.

**C. Memory (the "when")**
- **Experience replay** — when replay fires, animate the swept trajectory over the map (forward = magenta look-ahead,
  reverse = cyan credit-assignment), with a small event ticker.
- **Slow-weight consolidation** — a slow indicator of episodic → network-weight transfer over the lifetime;
  optionally a recent-vs-remote memory-strength readout.

Every panel gets a one-line plain-language caption ("Spatial code: the agent's learned coordinate system") and a
toggle. Keep it legible — no more than ~6–8 panels at once; let the user pick.

---

## 6. Visual design

Modern, dark, "mission-control for an AI" aesthetic — a clean scientific instrument, not a busy game HUD.
- Central 3-D viewport with smooth, appealing rendering (soft shadows, gentle ambient occlusion, readable materials);
  a free-orbit camera + a "follow the agent" camera + a top-down map camera.
- Panels docked in a tidy grid with generous spacing, subtle motion, and a restrained, **colorblind-safe** palette
  (one accent per module, used consistently everywhere that module appears). Smooth animation, no jitter.
- Typography: one clean sans; big legible numbers on the meters. A quiet graticule background ties it together.
- It should look like something you'd put on a conference keynote slide — calm, precise, alive.

---

## 7. Controls & UX

- Menu: choose environment, choose agent config (all-on, or a preset), Start.
- In-scene: **play / pause / step**, **speed** (0.25×–8×), camera modes, panel toggles, and a **per-module ablation
  switch** (turn curiosity / auto-goals / world model / planning / reward-momentum / a spatial module on/off and
  watch behavior change — this doubles as the demo's proof of fidelity).
- Optional: click a unit in a spatial panel to spotlight its field in 3-D; click a floor tile to hand the agent a
  goal there and watch it plan to it.
- A subtle "record" button that dumps a short clip / screenshot for sharing.

---

## 8. Architecture (recommended; use your judgment)

- **Agent process:** the real Python/PyTorch modules running the §4 loop, emitting a compact **state packet each
  tick**: agent pose; per-module spatial activations; active localization units; heading estimate; pose belief +
  uncertainty; curiosity/learning-progress field (sampled grid); current goal; list of rollout trajectories
  (polylines); self/external prediction-error; reward-momentum scalar + history; resource levels; replay events.
  Define this as a **versioned schema** (JSON for clarity, or packed binary if you need bandwidth).
- **World + renderer:** pick the stack you can make most beautiful and ship as a Windows `.exe`. Options, in rough
  order of "reuses the Python agent most directly":
  1. **All-Python** — 3-D via Panda3D / Ursina, dashboards via Dear PyGui / imgui, packaged with PyInstaller.
     Simplest data path (in-process), one language.
  2. **Godot 4** for world + dashboard (native Windows export) + the Python agent as a subprocess over a local
     socket / shared memory. Best-looking with least renderer friction.
  3. **Web tech** (Three.js + a bundled Python local server) wrapped with Tauri / Electron → `.exe`.
  Whatever you choose: **decouple tick rate from frame rate**, keep the state-packet interface clean, and make the
  build **one command** that produces the `.exe`.
- Ship: the `.exe`, source, `build.(bat|ps1)`, a README (how to run, what each panel means, how to add worlds), and
  the module-fidelity test.

---

## 9. Milestones (ship each as a runnable build)

1. **M1 — Body in a world.** The three environments load from data; the agent locomotes via the real model-based
   controller; menu + cameras + play/pause; produces a Windows `.exe`.
2. **M2 — The "where."** Live spatial panels (spatial code, localization units, heading, obstacle/landmark detectors,
   pose belief + uncertainty), faithful to the real modules; watch the representation build as it explores.
3. **M3 — The "why."** The autonomy modules wired and visualized: curiosity/learning-progress map, self-generated
   goal, rollout ghost-paths, self/external prediction-error meter, reward-momentum tint/timeline, resource bars —
   plus per-module ablation toggles that visibly change behavior.
4. **M4 — The "when" + polish.** Replay / consolidation viz, the open-world day/night + resources, visual polish,
   recording, README, and the fidelity test. Final `.exe`.

---

## 10. Acceptance criteria

- Double-clicking the `.exe` on a clean Windows machine opens the app; the user picks a maze or the open world and
  watches an agent that **explores on its own, sets and pursues its own goals, plans around obstacles by rolling out
  its world model, keeps its resource variables regulated, and shows reward-momentum shifts** — with every on-screen
  value traceable to a real module's output.
- Ablating any autonomy module **visibly** removes its contribution (the built-in dissociation).
- It is genuinely beautiful and legible — a thing worth showing off — **and** every signal is true.

Build it so that when it runs, we're not watching a scripted game character. We're watching an autonomous AI agent
explore a world, and seeing what its network is computing while it does. Make it honest. Make it stunning.
