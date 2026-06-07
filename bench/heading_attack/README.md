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
