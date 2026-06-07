# Heading-Recovery Attack (tree-structured)

## Why this folder
The whole Stage-1 OOD problem collapsed to ONE bottleneck (rigorously isolated): **estimate the agent's
absolute heading**. With oracle heading, dumb dead-reckon tracks a 129x129 traversal near-perfectly
(GLOBAL 0.38); with drifting heading it's catastrophic (31.6). So: solve heading -> solve Stage-1.

## The theoretical structure of the heading problem
heading_from_start theta. Available signals:
- **command integration** (turn history): exact in the COMMAND frame, but TRUE heading drifts from it by
  accumulated rot_noise (~0.09/step random walk -> ~4.5 rad over a long episode -> useless quadrant ref).
- **Manhattan grid** (freq-4 of omni view): TRUE heading **mod 90deg**, DRIFT-FREE, ~20deg noise. The win.
- **the missing 2 bits = the QUADRANT** (which of N/E/S/W). 4-fold symmetric locally -> not in a single view.

So the entire residual = **resolve the discrete 4-fold quadrant** (continuous part is solved drift-free by grid).

## Branch status (update as we go; failures keep their reason)
| branch | idea | status | result / why |
|--------|------|--------|--------------|
| b01 cmd+grid fusion | complementary filter (R6-A) | DONE | partial: mod-90 ok, quadrant drifts/flips |
| b02 learned estimator | recurrent fuse cmd+grid+obs (round16) | DONE | partial: 30deg(n6)->67deg(n64), quadrant-limited |
| b03 grid-canon + loop-closure quadrant | views drift-free mod-90; cross-correlate same-place visits -> relative rotation is a multiple of 90 (DISCRETE, robust) -> propagate absolute quadrant from start | TESTING | b03a kernel first |
| b04 asymmetric-structure quadrant | resolve quadrant from the maze instance's global asymmetry (window/map) | TODO | |
| b05 multi-hypothesis | track 4 quadrant hypotheses, prune by consistency | TODO | |
| b06 obs-anchored denoising | idea-A done right (train with injected drift + obs-anchored recovery) | TODO | |

## Method
Each branch: hypothesis -> minimal kernel test (CPU/substrate, fast) -> if kernel works, scale (FASRC) ->
record result + WHY (success/fail) here -> spawn next branches. Push until theoretically optimal (or prove
the information limit).

## b03 results
- **b03a kernel (cross-corr same-cell relative heading)**: median ~21deg, size-invariant, PLACE-SPECIFIC
  (diff-cell ~88-96deg = random). BUT mean ~44deg; **~38% of pairs err by ~a multiple of 90** = the place is
  near-4-fold-symmetric in appearance, so cross-corr picks the wrong quadrant ~38% of the time. So per-closure
  quadrant accuracy ~62%. -> partial; needs voting/propagation over many closures + grid (mod-90) + cmd prior.
- **b03b (full propagation: cmd prior + grid mod-90 + cross-corr loop-closure re-anchor)**: TESTING.
- **b03b (full: cmd + grid-snap + cross-corr greedy)**: **FAILED** — 85-93deg, WORSE than cmd-only (37-77)
  and b02 (30-67). WHY: grid + cross-corr are both mod-90 (quadrant-ambiguous); greedily snapping to them
  DESTROYS the quadrant that cmd-continuity maintained, and the 38% wrong cross-corr closures inject 90deg
  errors -> collapses to mod-90 ambiguity (~90deg). LESSON: don't greedily snap mod-90 signals; the quadrant
  must be carried by continuity (cmd) and only GENTLY corrected by high-confidence closures.

## Key theoretical distinction (why heading is still worth pushing, unlike the gauge)
- The **gauge sqrt-drift** (global position) is TRULY information-unrecoverable (relative measurements + one
  anchor). Don't fight it.
- The **heading quadrant** is RECOVERABLE IN PRINCIPLE: the maze instance is asymmetric, so the global
  structure DETERMINES the absolute orientation. It's an INFERENCE problem (hard), not an impossibility.
  -> worth attacking with global structure inference, not just local cues.

## Current best & next branches
- BEST so far: **b02 learned full-episode estimator, 30deg(n6) -> 67deg(n64)** (grows: quadrant drifts faster
  than the learned net re-anchors over long traversal).
- Next: b03c LEARNED rotation matcher (>62% vs raw cross-corr) feeding GENTLE closure; b04 learned global
  structure -> quadrant (window/map); b05 multi-hypothesis (4 quadrant tracks, prune); b06 obs-anchored
  denoising training. Honest expectation: quadrant is hard; capped by rotation-aliasing (places that look
  4-fold-symmetric). Push to find the practical best.

## b03c BREAKTHROUGH
- **b03c learned quadrant matcher** (geo+lm cross-corr features -> 4-class relative quadrant): **95-96%
  accuracy, SIZE-INVARIANT** (raw cross-corr 80%, earlier 62%). The landmark GEOMETRY (not IDs -> transfers
  OOD, flat across sizes) breaks the rotation symmetry the raw geo cross-corr missed. -> RELIABLE quadrant
  closures, finally. This is what b03b lacked (62%+greedy). Next b03d: integrate cmd(continuity) + grid(mod-90
  fine) + 95% matcher (quadrant re-anchor, GENTLE + voting, NOT greedy) -> resolve heading.

## b03d/e FINAL (after fixing a Gauss-Newton SIGN bug that had broken ALL integration attempts)
The persistent ~90deg across b03b/d/e was a GN sign bug (rhs must be -w*r). FIXED: on a SYNTHETIC angular
pose-graph with oracle far-apart closures -> 12deg (solver correct). But on the MAZE, even with ORACLE
closures + the fixed solver, heading recovery is UNRELIABLE / HIGH-VARIANCE (20deg on some mazes, 90-144deg
on others) and the grid per-step constraint HURTS (frame/noise). Two fundamental reasons surfaced:
  1. **Winding ambiguity**: total heading drift ~0.09*sqrt(T); when it exceeds pi (T~1200 -> n~30 -> ~61x61),
     RELATIVE closures (mod 2pi) can't recover the absolute winding -> heading unrecoverable beyond that size.
  2. **Gauge-like drift**: relative closures only make heading consistent; the absolute drifts with closure-
     distance to the start anchor. Closures back to start are sparse (DFS), so it drifts.
Plus grid noise (~20deg) defeats per-step quadrant unwrapping (b03-grid-unwrap: ~90deg, flips persist).

## VERDICT on the heading attack (exhaustive: b01,b02,b03a-e, grid-unwrap, oracle/learned, ~10 approaches)
- REAL component: **b03c 95% relative-quadrant matcher** (landmark-geometry, OOD-transferable).
- The grid gives DRIFT-FREE heading mod-90 (~20deg noise).
- **Absolute heading is FUNDAMENTALLY BOUNDED**: (a) winding ambiguity for total-drift>pi (size set by
  rot_noise; ~61x61 at 0.09), (b) gauge drift to the start anchor, (c) grid noise defeats unwrapping. No
  reliable cross-size absolute-heading solve was found.
- BEST consistent estimator: **b02 learned full-episode, ~30deg(n6) -> 67deg(n64)** (mediocre, degrading).
- So GLOBAL state-tracking (which needs heading) is fundamentally OOD-bounded; LOCAL is size-invariant;
  NAVIGATION is topological and does not need absolute heading (the practical out).

## ★ CORRECTED FINAL VERDICT (b04-b09) — the previous verdict was right about the START frame but MISFRAMED it
Prompted by the question "is it really fundamental, or could more data/compute push further?", I re-attacked
with the right frame and an estimator-free information analysis. Result: a clean, multiply-confirmed decomposition.

**A bug first:** `_grid_heading` is REFLECTED (gh ~ HALF - ang%HALF), so it ANTI-rotates with the command.
Harmless for the relative matcher (b03c), FATAL for absolute tracking. Negate it (`gh = -_grid_heading`). Also:
the "~20deg grid noise" in the old verdict was the reflection artifact — the TRUE fine-heading noise is ~5deg.

**The decomposition (heading rel walls = fine mod-90 + 90*quadrant):**
- **Fine heading mod-90 (vs the WALLS): drift-free ~5deg, SIZE-INVARIANT to 259x259** (b06). The walls are an
  omnipresent global landmark -> this is allothetic, not path-integrated -> no drift, no size dependence.
- **The QUADRANT (2 bits): gauge-bound, GROWS with size, and is the ONLY hard part.**
  - b05 (estimator-free information floor = effective resistance to the start anchor): EXPLORE 13->26->42deg
    (n=6->12->24); DENSE-revisit 9->16->25deg. Grows because it's referenced to a SINGLE anchor and the
    closure graph's resistance to it grows with maze diameter. This is the optimal floor — no estimator beats it.
  - b07 (re-anchor the quadrant by loop closure): does NOT fix it — the anchors are themselves quadrant-
    uncertain, so re-anchoring spreads errors instead of removing them. (Confirms b05: it's information, not estimator.)
  - b08 (can a LEARNED net read the absolute quadrant from one view? = the "more data/compute" test):
    train acc 95.8% (pure per-maze memorization) but **balanced OOD acc ~28-30%, declining to chance(25%)**.
    The absolute quadrant is NOT in the local views — the maze is (near-)exactly 4-fold rotationally symmetric.
    => NOT a data/compute limitation; the information is physically absent.
- **The whole blocker is the 4-fold symmetry.** b09: add ONE omnipresent orientation cue (a compass/sun/
  direction-colored walls), even 10%-wrong, and FULL absolute heading becomes SIZE-INVARIANT: p=0 -> 12deg,
  p=0.1 -> 23deg, DEAD FLAT 13x13 -> 259x259. No cue -> drifts ~85deg.

**Answer to "can we go forever (math/physics/neuro)?":**
- YES, forever & size-invariantly, for everything anchored to a GLOBAL reference: fine heading mod-90 (walls),
  local windowed metric, topology. These are at their information floor (~5deg) and flat to 259x259.
- NO for the single global symmetry label (the quadrant) + global metric position: gauge-bound (grow with
  distance-from-the-one-anchor), AND information-theoretically unrecoverable from local views in a symmetric
  maze. This is NOT fixable by data/compute/architecture. It IS fixable by ONE featural global cue.
- This is gauge theory (global-field observables bounded; single-point-referenced observables integrate noise)
  AND classical neuro: head-direction cells path-integrate (drift) + re-anchor to allothetic cues; and the
  residual symmetry is exactly Cheng-1986's GEOMETRIC MODULE rotational error (animals make 180deg errors in a
  2-fold-symmetric box, cannot break it with geometry alone, need a featural cue). Our 4-fold maze = same law.

=> SPOTLIGHT FRAME: egocentric state in a (symmetric) maze splits provably into a globally-anchored
SIZE-INVARIANT part (recoverable to the info floor at ANY size) + a single global symmetry label that is
gauge-bound & locally-unobservable (collapsed by one featural cue). Baselines that integrate ONE global frame
drift on the second part; a tracker that locks the recoverable decomposition (grid mod-90 + topology + local
metric) and treats the symmetry label as a settable latent is size-invariant by construction. Three independent
confirmations (information floor, online tracker, learnability) + neuro grounding + a benchmark with a symmetry knob.
