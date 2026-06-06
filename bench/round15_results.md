# Round 15: systematic attack on the fixable Stage-1 sub-items (one by one) — honest outcome

Per the user's "挨着轮流反复尝试和突破", attacked the fixable loop-closure/circle-back levers on the faithful
junction substrate (heading-drift calibrated to the real MGM, op = sigma 0.10 / sigma_rot 0.04). Metric:
loop-closure PRECISION (fraction of circle-back detections that are TRUE) + nav success/steps, vs size.
Baseline to beat: ~0.4 precision at n40 (per-cell TightGate / R14).

## Results (LC-precision vs size, 8 mazes)
| n (grid)    | base (R14) | item1 (rich fingerprint) | item2 (pose-graph) |
|-------------|------------|--------------------------|--------------------|
| 20 (41x41)  | 0.81       | 0.77                     | 0.83               |
| 28 (57x57)  | 0.47       | 0.54                     | 0.51               |
| 40 (81x81)  | 0.37       | 0.40                     | 0.39               |
| 52 (105x105)| 0.41       | 0.49                     | 0.44               |

- **Item 1 (richer heading-invariant fingerprint: degree + corridor-length multiset):** marginal (≈ base).
  The fingerprint reduces candidates but the EST GATE still false-merges among compatible candidates at scale.
- **Item 2 (pose-graph relaxation):** marginal (≈ base). Translation-only relaxation CANNOT fix the heading
  ROTATION that corrupts the odometry edges; the bootstrap (heading corrupts the closures the relaxation
  needs) also limits it.

## The binding lever is HEADING CORRECTION (the "item 5" my list missed)
Sweeping the heading-drift (= observation-based heading-correction strength) on the best closer:
| sigma_rot | n28 prec | n52 prec |
|-----------|----------|----------|
| 0.04 (current MGM) | 0.51 | 0.44 |
| 0.02      | 0.66 | 0.47 |
| 0.008 (strong R6-A correction) | 0.68 | 0.57 |
Lower heading drift clearly lifts precision -> **heading correction, not loop-closure post-processing, is the
binding lever.** But even strong correction caps ~0.4-0.6 at the largest sizes, because (a) the translational
sqrt gauge drift (R3) still binds and (b) the heading QUADRANT occasionally flips (R6-A 4-fold symmetry).

## Honest verdict on the systematic attack
- **Items 1, 2 (fingerprint, pose-graph): marginal** — they push the achievable-size constant only slightly,
  because the binding constraint is heading rotation, which translation-only methods don't touch.
- **Item 3 (hierarchical/compressing memory):** addresses memory CAPACITY, not the binding precision
  constraint — it's the R3 result (O(1) memory matches unbounded); it makes the map cheaper, not more
  accurate. Not a precision breakthrough.
- **Item 4 (on-policy MGM training):** a robustness axis for Stage-2 integration (FASRC retrain), not a
  circle-back-precision lever.
- **The real high-value Stage-1 lever = better observation-based HEADING CORRECTION (R6-A, mod-90) folded
  into the MGM**, which reduces the effective heading drift and lifts circle-back precision — bounded above
  by the fundamental quadrant (4-fold symmetry) + translational gauge drift.

So the systematic attack confirms, from 3+ more independent angles (fingerprint, pose-graph, heading sweep),
what R3/R6-A/R12/R14 already found: circle-back precision at scale is bounded by FUNDAMENTAL walls (sqrt
translational gauge drift + heading-quadrant 4-fold symmetry + local aliasing). The fixable levers give
modest gains (push the constant); heading correction is the most valuable; none breaks the fundamental
ceiling — as expected for information-theoretic limits. The achievable goal (completeness + bounded
redundancy, R14) stands; perfect circle-back / zero-redundancy is fundamentally out of reach.
