# TrackMaze — a controllable egocentric maze for **state-tracking under OOD**

**What it is.** A 2.5D, egocentric, *partial-observation* maze benchmark for the question:
> Can a model learn to **track its latent spatial state** (where am I + the map) from a first-person
> stream, and **generalize from small mazes to large ones** (the OOD axis)?

Existing benches sit at two extremes — **MiniGrid** (2D, cheap, but the memory tasks are toy) and
**Memory Maze** (3D, on-target, but heavy/MuJoCo, 100 GB, and couples localization with object memory).
TrackMaze aims at the empty middle: **finely controllable difficulty**, **pure state-tracking isolated**,
the **drift ↔ correction** mechanism instrumented, and **cheap to iterate** (2.5D raycast from a 2D grid).

## Play the demo
`demo.html` is a self-contained, dependency-free playable prototype.
- **Left** = what the agent sees (egocentric 2.5D raycast). **Right** = the ground truth it must *infer*.
- Walk with `W/A/S/D` (or arrows), reach the green pillar. Colored walls = **landmarks** (re-anchor cues).
- Sliders: **maze size** (the OOD axis), **landmark density** (how much re-anchoring is possible — set 0 to
  *feel the drift*), **observation ambiguity** (aliasing → loop-closure is harder).
- Toggles: **localization challenge** (hide your dot → you must track yourself), **fog of war** (partial obs).

Open it locally: `open demo.html` — or play the live version (GitHub Pages, if enabled).

## Design (what the full benchmark will be)
- **Observation:** egocentric, partial (first-person 2.5D view + heading).
- **Controllable difficulty:** size `S` (OOD axis: train small → test large) · landmark density `ρ_L`
  · observation ambiguity `α` · episode/path length · loopiness (tree vs loopy).
- **Ground truth provided:** position, heading, full map, visited cells, **re-anchor events** — so the
  drift/correction metrics are computable.
- **Tasks:** (T1) **localization** (track your position — the purest state-tracking) · (T2) **navigation**
  (reach a goal under partial obs) · (T3, opt) map reconstruction.
- **Metrics:** localization **error vs step** (drift curve) + **error vs steps-since-re-anchor** (correction)
  · navigation success + SPL-vs-optimal-explorer · a state **probe** (decode position/map from the latent)
  · **the headline: the generalization curve** — every metric vs maze size, showing *where each method collapses*.
- **Baselines (to ship):** GRU/LSTM · transformer · SSM (Mamba) · segment-recurrent hybrid · factorized
  (TEM-style) tracker · optimal-explorer / SLAM oracle (ceiling). Plus a human baseline.

## Status
`2026-06-05` — **playable demo prototype** (`demo.html`). Next: package as a gym/dm_env environment with
ground-truth export, the metrics + state-probe harness, and the baseline suite + collapse curves.

## Repo
```
demo.html   self-contained playable 2.5D demo (this is what you click)
README.md   this file
```
