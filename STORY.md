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

## 5.5 The behavioral payoff: structure beats end-to-end for NAVIGATION too (R13, `bench/round13_results.md`)
The §5 result is about ESTIMATION (predict heading). Does it carry to closed-loop BEHAVIOR — a learned policy that
actually *walks* the maze? We train a navigation policy (find the exit) by behaviour-cloning a frame-consistent
frontier-DFS teacher on SMALL mazes (11–15px) and evaluate it closed-loop, egocentric-only, on OOD-LARGER mazes
(up to 81px, ~6×). The only thing we vary is the inductive bias (`bench/nav_bc.py`, `nav_ood_figure.png`):
- **GRU, drifting frame (end-to-end):** canonicalize observations by the integrated command heading → **~0.15
  reach in-distribution and 0% from 2× size onward** (the frame rotates with the unobservable heading drift; the
  learned strategy doesn't transfer). Pure-RL (PPO) couldn't even learn it (sparse-reward wall).
- **GRU, grid-corrected frame:** canonicalize by the drift-free allothetic grid-mod-90 cue (the §3 wall cue) →
  **~0.8 reach in-distribution, degrades GRACEFULLY to ~0.25 at 81px (~6× train size)**, 2–4× more step-efficient.
- **+ spatial memory on drift-free topological coords (OURS):** ≈ the grid-corrected GRU on exploration (the
  memory lever WASHES OUT once the frame is corrected and training is sufficient — an honest finding), but it is
  the clear winner on maze HOMING, which genuinely requires the cognitive map (ours 0.50 vs gru_corr 0.32 vs
  gru_cmd 0.17 in-distribution).
- **Transformer, same drift-free obs:** collapses by 2× — *capacity is not the lever; the right recurrent
  structure is.*
- **+ allothetic GLOBAL compass cue (resolves the Z₄ quadrant, §7):** the only model that **sustains reach at the
  most extreme size (~6×: 0.40 vs 0.17–0.21)**, where even the grid-corrected frame's residual quadrant drift
  finally bites — the §7 "an allothetic orientation cue resolves the global frame at scale" principle realized in
  behaviour. (Honest: it does NOT give a flat size-invariant line — the exploration task difficulty itself scales
  with size — and costs slightly in-distribution; a suggestive, not dramatic, extreme-OOD gain.)
The rigorous controlled arm is GRU-drifting vs GRU-grid-corrected (identical architecture, only the obs-frame
canonicalization differs) → the allothetic frame correction is causally THE lever. (`nav_ood_strong.png`.)
This is the **same dissociation as §5, now in behaviour**: a drift-free allothetic frame (plus a growing spatial
memory) is what makes OOD-size navigation possible; end-to-end integration of a drifting frame fails
catastrophically — exactly what the bounded-error theory predicts (drifting state = unbounded with size;
allothetically-anchored state = bounded/size-invariant). Honest scope: absolute reach is moderate (one-shot BC of
a long-horizon explorer is hard) and even the corrected frame eventually fails at extreme OOD (the quadrant-drift
limit of §3–§4); the result is the **dissociation and the graceful-vs-catastrophic degradation**, not a perfect
navigator. Navigation is *topological*, so it sidesteps the global-frame metric estimator's brittleness (§6).

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
   **17px 1.8° · 33px 3.1° · 49px 3.6°** — flat, size-invariant.
   - **≥65px wall — NOW CLOSED (2026-06-08, `scaling_result.txt` / `scaling_fixed.png`).** The borderline
     brittleness (one 65px run at ~50° despite 95% q-acc) was NOT a solver problem: the root cause was
     **closure-density collapse** — the closure budget was held FIXED while the cell count grows ~n², so
     closures/cell fell below the phase-diagram 2D-rigidity threshold (~40/cell) and the Z₄ field became
     **under-determined** (no solver — BP, LS, or spectral — can recover an under-constrained field; that is
     exactly why q-acc 26% coexisted with closure-acc 100%). Fix = build **~40 closures per cell** from the
     abundant same-cell revisit pairs (short lever arms). Result: **gauge-invariant heading flat & low (4–12°),
     q-acc ≥88%, closure-acc 100% from 17px to 129px** — the wall is gone. (Recorded negative: the continuous-LS
     Z₄ solver is a genuine dead-end — 25–32% q-acc even at n=8; the Z₄→real relaxation mishandles the 180°
     residue. A cell-graph reduction is a 100–400× speed optimization for going past 129px, not a correctness fix.)

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
