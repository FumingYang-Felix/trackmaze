# TrackMaze — a controllable egocentric maze for **state-tracking under OOD**

**What it is.** A 2.5D, egocentric, *partial-observation* maze benchmark for the question:
> Can a model learn to **track its latent spatial state** (where am I + the map) from a first-person
> stream, and **generalize from small mazes to large ones** (the OOD axis)?

Existing benches sit at two extremes — **MiniGrid** (2D, cheap, but the memory tasks are toy) and
**Memory Maze** (3D, on-target, but heavy/MuJoCo, 100 GB, and couples localization with object memory).
TrackMaze aims at the empty middle: **finely controllable difficulty**, **pure state-tracking isolated**,
the **drift ↔ correction** mechanism instrumented, and **cheap to iterate** (2.5D raycast from a 2D grid).

> ### 📖 The research story → [`STORY.md`](STORY.md)
> Read this first if you're following the project. It walks the whole journey step by step — the dead-ends and
> *why* they failed, the confound we caught and corrected, the gauge reframe — to the spotlight result:
> **OOD egocentric self-localization is a *phase transition* in environment topology × coverage** (we map it);
> **inductive bias (grid-anchoring) beats end-to-end** Transformers/RNNs on OOD size; and we **realize the
> recoverable phase end-to-end** with a realistic allothetic sensor — unified by one principle (orientation in
> open environments needs an allothetic reference, at both local and global scales; the geometric-module /
> Cheng-1986 law generalized). The formal spine is [`bench/heading_attack/THEORY.md`](bench/heading_attack/THEORY.md);
> the spotlight figures are `bench/deployable/phase_*.png`.

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

## The benchmark code (`bench/`)
The demo geometry, ported to Python as a trainable env + ground-truth export + metrics.
```
bench/env.py            TrackMazeEnv: maze gen + raycast egocentric obs + full GT + the 3 difficulty knobs
                        + the drift source (unobservable odometry noise) the agent must correct
bench/generate.py       scripted-explorer trajectory datasets (obs stream + actions + GT, incl. the
                        egocentric displacement-from-start target for path integration)
bench/metrics.py        drift curve · correction curve (err vs steps-since-loop-closure) · SPL ·
                        generalization (any metric vs maze size) · dead_reckon open-loop reference
bench/run_harness.py    sanity: open-loop dead-reckoning reproduces drift, growing + worse on larger mazes
bench/train_baseline.py first learned baseline (GRU tracker) + the with/without-landmark ablation that IS
                        the bench's question: does seeing landmarks let the tracker correct its drift?
```
Design choices that make it a *clean* state-tracking probe:
- The agent observes only `rays + landmark-id-per-ray + its own last command` — **never** its true
  pose. Drift is injected as **unobservable, persistent heading + translation odometry noise**, so the
  latent genuinely has to be tracked (and can only be corrected by recognizing landmarks).
- Landmark **ids are bounded** (ambiguity ≥ 1) so they stay in-range when you test on a *larger* maze
  than you trained on; re-anchoring therefore must work by **matching / loop-closure**, not by reading a
  global id. Re-anchor events are defined by **trajectory loop-closure** (vocabulary-independent).
- The headline metric is the **generalization curve**: every quantity vs maze size — *where each method
  collapses* as the OOD gap widens.

```bash
cd bench && python run_harness.py        # validate env+GT+metrics (no torch)
python train_baseline.py                 # GRU baseline + landmark ablation (torch, a few min)
```

## Status
`2026-06-05` — env + ground-truth + metrics + first baseline shipped (`bench/`); validated correctable drift.

`2026-06-07` — **research arc complete (see [`STORY.md`](STORY.md))**:
- **Theory:** a predictive **phase diagram** of when OOD global self-localization is size-invariant —
  recoverable only in **loopy/2-D-connected** environments with enough coverage; **tree-mazes are
  fundamentally unrecoverable at scale** (Pólya / lower-critical-dimension). Spine:
  `bench/heading_attack/THEORY.md`; figures `bench/deployable/phase_*.png`.
- **Caught + corrected a confound** (a spinning scripted explorer faked early "size-invariant" wins) — honest
  re-eval drove the whole reframe.
- **Method:** explicit **grid-anchoring** (fine heading vs the walls, drift-free) + quadrant tracking is
  **flat ~2°** on OOD global heading where GRU/Transformer drift to 21–58°; ablation shows the win is the
  **inductive bias**.
- **Deployable:** end-to-end global-heading recovery, **no oracle rotation**, **flat ~2–4° to 49×49** using a
  realistic **allothetic** sensor (distinct distant cues) — which also resolves the global symmetry (one cue,
  both scales). Larger sizes need the Z₄-sync back-end's iterations ∝ size and are currently brittle at the
  borderline (a convergence issue, not fundamental). `bench/deployable/deploy_slam.py`.
- **Honest limits:** tree-mazes unrecoverable (fundamental); appearance place-recognition under aliasing is the
  open sub-problem; navigation is topological and needs none of this.

Next: write the spotlight; harden the deployable place-recognition front-end; scale the Z₄-sync back-end
(iterations ∝ size) and/or a cell-graph reduction.

## Repo
```
demo.html                       self-contained playable 2.5D demo (this is what you click)
README.md                       this file
STORY.md                        the research narrative + spotlight (read this to follow the project)
bench/                          the trainable benchmark (env + GT + metrics + baselines)
bench/round*_results.md         the early rounds R2..R15 (incl. the confound + correction)
bench/heading_attack/           the heading attack b01..b17 + THEORY.md (gauge hierarchy + phase transition)
bench/deployable/               phase diagram, method-vs-baselines, and the working deployable SLAM
bench/deployable/phase_*.png    the spotlight phase-diagram figures
```
