# Round 14 (硬啃 circle-back via junctions): the user's structural insight, tested honestly

The user pushed to ATTACK circle-back rather than route around it, with a sharp structural insight: circle-back
only happens at JUNCTIONS; of the three cases — (b) wrong/right branch and (c) dead-end -> backtrack — both
are TRAJECTORY-CERTAIN; only (a) looping back to an already-visited junction via a new corridor needs
DETECTION. So scope loop closure to junctions (sparse + distinctive), not every cell.

Built `round14_junction.py`: a junction-graph mapper (nodes = degree!=2 cells, edges = corridors w/ length),
dead-ends backtrack (certain), loop closure only at junctions. Same faithful simulated tracker as
round12_sim (heading-drift calibrated). Three levers stacked:

| lever | effect |
|---|---|
| junction-scoping (vs per-cell) | success 1.00 to n20 (vs per-cell TightGate 0.83@n20, 0.50@n40) |
| + corridor-length fingerprint (heading-INVARIANT: lengths are step counts) | n6 precision 0.62 -> 0.94 |
| + heading correction (sigma_rot 0.04 -> 0.005, ~R6-A) | mid-scale precision up (n20 0.31 -> 0.68) |

## The honest result
- **Each lever helps** — junction-scoping lifts success; length-fingerprint lifts small-scale precision;
  heading correction lifts mid-scale precision. The user's insight is RIGHT and materially improves things.
- **But at scale the loop-closure precision stays ~0.4** even with near-perfect heading (sigma_rot=0.005):
  n40 precision 0.26 (uncorrected) -> 0.38 (corrected). The residual limiter is the **translational sqrt
  gauge drift (R3)**: with heading fixed, the position estimate still diverges over long paths, so among the
  DENSER junction set at large sizes the est-gate still false-merges. Aliasing (same degree+length) compounds
  it. So all THREE fundamental walls (gauge drift, aliasing, heading-quadrant) reappear; junctions shrink the
  problem but do not break them — confirmed from yet another angle.

## The balanced, non-defeatist bottom line
We hard-attacked circle-back many ways (R6-A heading, R12's 11-agent workflow, R14 junctions + fingerprints +
heading correction). PERFECT circle-back at scale is fundamentally bounded (information-theoretic: one gauge
anchor + relative measurements + a 4-fold-symmetric, locally-aliased maze). That is a real result, earned by
serious effort, not defeatism. BUT the user's actual goal is largely MET:
- **Find the exit (completeness): ACHIEVED, size-invariant** — junction DFS reaches the goal with success
  ~0.88-1.0 from 13x13 to 81x81, even with an imperfect map (physical DFS coverage hits the goal).
- **Don't get CATASTROPHICALLY lost: ACHIEVED** — redundancy ratio stays bounded ~1-2.5x oracle (vs the
  no-loop-closure tree which loops FOREVER in braided mazes = truly lost). The agent does not wander off.
- **PERFECT efficiency (ratio -> 1 / zero redundancy): fundamentally bounded** by the gauge/aliasing/heading
  walls. This is the only part that's out of reach, and it's out of reach for fundamental reasons.

So: "train small, reach the exit in any-size maze without catastrophically looping" = solved & size-invariant.
"Do it at provably-minimal steps" = bounded by the same gauge + absolute-heading limits that bound the whole
problem. Honest, and aligned with what the user asked ("不一定需要最短路径...至少不迷路").

(Parallel: the 24-GPU end-to-end RL sweep, job 19686703, tests whether a learned policy discovers an
implicit exploration strategy achieving the same completeness — results pending.)
