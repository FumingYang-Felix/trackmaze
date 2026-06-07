# TrackMaze — the research story: how we got here, and the spotlight

This document is the narrative spine of the project for anyone following the repo. It tells, step by step,
how a simple benchmark question turned into a clean **theory of when OOD egocentric self-localization is
possible**, a **method that beats end-to-end baselines**, and a **working deployable estimator** — plus the
dead-ends and *why* they failed (those were as informative as the wins). File pointers are given throughout so
you can verify every claim against code/results.

> TL;DR of the spotlight. OOD-size egocentric state tracking is **not one problem with one limit**. It factors
> into (i) a **size-invariant, globally-anchored part** (fine heading vs the walls, local metric, topology),
> and (ii) a **global frame** (absolute heading-quadrant + position) whose recoverability is a **phase
> transition in environment topology × coverage** (we map it), with a residual **2-bit symmetry** that is
> irreducible without an allothetic cue. Standard RNN/Transformers integrate a single global frame and **drift
> on OOD size**; a tracker that **locks the recoverable factorization** is size-invariant by construction, and
> with one **allothetic orientation cue** (sun/skyline/distant landmark) the global frame is recoverable too —
> we demonstrate it end-to-end. The same allothetic cue resolves *both* the local closure-rotation and the
> global symmetry: the geometric-module / Cheng-1986 law, generalized to two scales.

---

## 0. The question and the benchmark
**Can a model track its latent spatial state from a first-person stream and generalize from small mazes to
large ones?** TrackMaze is a controllable 2.5D egocentric maze with unobservable odometry+heading noise (so the
state genuinely must be tracked and corrected), and knobs for size, landmark density, ambiguity, and
**loopiness** (tree ↔ loopy). See `README.md` and `bench/env.py`. The headline axis is OOD **size** (train
small → test large).

---

## 1. The LOCAL/GLOBAL split, and the real wall (rounds R2–R7, `bench/round*_results.md`)
- **R2.** Continuous-observation integration makes **LOCAL** state (windowed relative displacement)
  **size-invariant** — achieved, flat. **GLOBAL** metric state drifts. This split organizes everything that
  follows.
- **R3.** We refuted our own "memory is the bottleneck" idea: a constant-size importance-weighted pose-graph
  matches the full one at the **√ single-anchor gauge floor**. Memory is not the wall (this is classical SLAM
  sparsification). The √ growth is a **gauge** limit, not a memory limit.
- **R4–R5.** The real wall is **vocabulary-independent place recognition under local aliasing**: even an oracle
  local-appearance descriptor tops out ~0.65 top-1 precision and **drops with size** — the maze is locally
  self-similar. Unique landmark IDs match trivially but don't transfer OOD (the *vocabulary trap*).
- **R6–R7.** Breakthrough: **motion-constrained matching** (match only motion-reachable neighbors, not all
  places) is size-invariant. Reframed the task (user): **reach the exit without getting lost**, not shortest
  path. Completeness is size-invariant for free; optimality needs aliasing-robust loop closure.

## 2. A learned architecture — and an honest confound (R8–R13)
- Built **MGM** (`bench/arch_mgm.py`): rot/MLP encoder + GRU integration + motion-gated bounded memory +
  loop-closure gate. First results looked great: "size-invariant LOCAL tracking, beats GRU/Transformer to
  129×129."
- **THE CONFOUND.** Those runs used a scripted explorer that **spins in place** (~few distinct cells regardless
  of maze size). The agent never actually traversed large mazes. On *honest* DFS-traversal data, **MGM ≈ GRU**.
  We owned this (the lesson: *verify the trajectory data before reporting a result*). It reshaped the whole
  project: the apparent win was an artifact; the real question is the GLOBAL frame, which everyone drifts on.
- Diagnosis: with oracle heading even dumb dead-reckoning tracks large mazes → **everything reduces to heading
  recovery**.

## 3. The heading attack, and the gauge reframe (`bench/heading_attack/`, THEORY.md)
We attacked absolute heading exhaustively (`b01`–`b17`, `heading_attack/README.md`). Key turns:
- The **grid** (freq-4 Fourier phase of the omni view) gives heading **mod 90°, drift-free, ~5°,
  size-invariant** — it's *allothetic* (the walls are an omnipresent landmark). (A sign bug — `_grid_heading`
  is reflected — masked this; the old "~20°" was the artifact.) See `b06_wallframe.py`.
- The only missing piece is the **quadrant** (which of 4) — a discrete **Z₄ field**. We first (wrongly) called
  it gauge-bound from a *continuous* effective-resistance argument; the user pushed back ("is there really no
  way past gauge?"), and re-examination showed the discrete field is a **synchronization / 2D-Potts** problem,
  not a continuous one.
- A **learned classifier cannot read the absolute quadrant from one view** (`b08`): balanced OOD = chance
  (25%), declining with size, despite 95.8% train (memorization). The maze is (near-)exactly 4-fold symmetric →
  the global frame offset is **information-theoretically unobservable** from local views. One omnipresent cue
  (compass), even 10% noisy, collapses it (`b09`, flat to 259×259).

## 4. The phase transition — *when* the global frame is recoverable (the spotlight core)
Estimator-free analysis (effective resistance to the anchor on the honest visit graph = the BLUE variance no
estimator beats; `bench/deployable/phase_diagram.py`, figures `phase_diagram.png` / `phase_lines.png`):
- **Tree-maze (loop≈0, topologically 1-D):** the global-heading floor **grows with size** (85→326° as the maze
  grows) → unrecoverable at scale. Fundamental (Pólya recurrence / lower-critical-dimension; 1-D has no
  long-range order).
- **Loopy / 2-D-connected (loop ≳ 0.6) + enough loop closures:** the floor **saturates, flat with size** (~20–
  30°, below the 45° quadrant threshold) → **recoverable, size-invariant**.
- It needs **both** loop topology **and** coverage/closure-density (the active-SLAM axis). The global **2-bit
  symmetry** is an orthogonal knob (irreducible without a featural cue).

**So "can we beat gauge?" → it's a phase transition in (topology) × (coverage) × (symmetry), not a yes/no.**
This is the predictive map; everything else realizes points in it.

## 5. The method vs baselines — structure beats end-to-end (`bench/deployable/baselines_vs_method.py`)
Predict global heading, train small (loopy) → test OOD large. All models fit training (~7°), but:
- raw **GRU / Transformer drift on OOD size** (21–58°) — they never discover the grid-anchoring from raw rays;
- **GRU + grid-feature** (feed the freq-4 phase) drops to 13–26° → the win is the **inductive bias**, not
  hand-coding;
- **OURS** (explicit grid-anchoring + quadrant tracker): **flat ~2°**.
The right structure makes global heading size-invariant; end-to-end integration drifts. (`compare_result.txt`.)

## 6. The deployable method — closing the loop without oracles (`bench/deployable/`)
Goal: realize the recoverable phase with a real estimator (no oracle). The journey localized the bottleneck and
then cleared it:
1. **Place recognition** (find revisits without oracle): hand-crafted *and* **learned** descriptors cap ~0.4–
   0.5 precision (`learned_placerec.py`) — appearance aliasing is real; the **position gate** (R6-C) is the
   lever, but it needs position → chicken-and-egg.
2. **Back-end + gate work; the matcher is the wall.** Clean ablation (`FINDINGS_a_b.md`): chain quadrant
   99.7%; Z₄-sync recovers ~100% **given correct closures**; oracle-position gate → 100% same-cell. But the
   **relative-quadrant matcher** is only 55–65% on the loopy distribution (vs 95% on DFS), declining with size.
3. **Why:** a structural **tension** — recoverability needs loopy/open topology, but rotation-matching needs
   wall geometry → *the recoverable regime has the weakest local cues.*
4. **Fixes tried (user's plan: multi-view, then richer sensor):**
   - multi-view / sequence (grid-canonicalized, windowed): ~44% — the local signal is *confidently wrong*
     (open cells ≈4-fold symmetric); averaging stays wrong (`multiview_rot.py`).
   - richer **wall-bound** landmark types: ~43% — open mazes have no walls to host them (`rich_sensor.py`).
   - **free-standing beacons** (objects/pillars): local ~58–64% flat; **distinct distant cues (compass/
     skyline): 90–98%, flat with size** (`beacon_sensor.py`, `SENSOR_FINDINGS.md`).
5. **End-to-end works** (`deploy_slam.py`, no oracle rotation): grid-fine chain + same-cell closures whose
   relative quadrant comes from the **beacon constellation** (two non-obvious fixes: **grid-correct** the
   closure quadrant; **well-distribute** the closures so BP converges) + Z₄-sync. Global heading:
   **17px 1.8° · 33px 3.1° · 49px 3.6°** — flat, size-invariant. At ≥65px the BP back-end needs iterations ∝
   size: closure accuracy stays 100% and q-accuracy recovers (38%→95% with more iters), and 89px reaches 6.3°,
   but it's currently **brittle at the borderline** (one 65px run sat at ~50° despite 95% q-accuracy — a
   back-end convergence / reference-reconstruction sensitivity, not a fundamental limit). Hardening the back-end
   at scale (more iters, a cell-graph reduction, or a non-iterative solver) is the clear next step
   (`scaling_result.txt`).

## 7. The unifying principle (the deepest finding)
**Orientation in open/recoverable environments requires an allothetic reference; local geometry is insufficient
at BOTH scales** — the local closure-rotation (open cells ≈4-fold symmetric) and the global frame (the maze is
≈4-fold symmetric, `b08`). **One allothetic cue resolves both**: it supplies the closure rotation (90–98%,
§6) and collapses the global 2-bit (`b09`). This ties the local and global walls together and is the classic
**geometric-module / Cheng-1986** result (animals can't orient by geometry alone in a symmetric box; they need
a featural cue) **generalized to two scales**. Real open environments have such cues (sun, skyline, distant
landmarks); a corridor-maze has wall geometry instead but is non-recoverable for the global frame anyway.

## 8. Honest limits
- We did **not** "solve" unbounded OOD in general — in tree-mazes the global frame is provably unrecoverable at
  scale (§4), and that's fundamental.
- The deployable demo uses **oracle place-id** for *which* timesteps are the same cell; appearance place-rec
  under aliasing (finding the pairs) remains the genuinely hard, separate sub-problem (§6.1). Navigation,
  however, is **topological** and does not need the global frame at all.
- The allothetic-cue result assumes the environment provides such a cue (real open ones do; our maze needed a
  beacon layer to model it).

## 9. The spotlight, in one paragraph
We turn a benchmark question into a **predictive phase diagram** of OOD egocentric self-localization (order
parameter = global-frame variance; critical dimension d = 2; orthogonal symmetry transition), show that
**inductive bias (grid-anchoring) beats end-to-end** integration on OOD size, and **realize the recoverable
phase end-to-end** with a realistic allothetic sensor — unified by a single principle (allothetic orientation
is required at both local and global scales) with a clean neuroscience grounding. Not "we solved everything";
a **theory of when it's possible, why, and how**, with the method and the limits made explicit.

## Where to look
```
README.md                          the benchmark + demo
bench/heading_attack/THEORY.md     the gauge hierarchy + phase-transition theory (the formal spine)
bench/heading_attack/README.md     the heading attack log (b01..b17), incl. corrected verdicts
bench/deployable/README.md         items 1 (method vs baselines) + 2 (phase diagram)
bench/deployable/phase_*.png       the spotlight phase-diagram figures
bench/deployable/FINDINGS_a_b.md   deployable bottleneck localization (place-rec / matcher / back-end)
bench/deployable/SENSOR_FINDINGS.md  multi-view → richer sensor → the allothetic resolution
bench/deployable/deploy_slam.py    the working end-to-end deployable method (no oracle rotation)
bench/round*_results.md            the early rounds (R2..R15), incl. the confound and its correction
```

## Reproduce the headline results
```bash
cd bench
# phase diagram (the spotlight figure):
python deployable/phase_diagram.py --loops 0.0 0.3 0.6 0.9 --sizes 8 12 16 24
# structure beats baselines on OOD global heading:
python deployable/baselines_vs_method.py --loop 0.9 --sizes 6 12 20 32
# the working deployable method (no oracle rotation), flat ~2-4deg:
python deployable/deploy_slam.py --loop 0.9 --sizes 8 16 24
# (Mac: prefix with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2)
```
