# Attacking the matcher wall: multi-view → richer sensor → the allothetic resolution

The deployable bottleneck (FINDINGS_a_b.md) was the relative-quadrant matcher: 55-65% on the loopy
(recoverable) distribution, below the back-end's ~90% need, because open/loopy cells are locally
rotation-ambiguous. User: "先尝试多视角/序列旋转估计器，如果实在不行就添加更丰富的传感器." Both tried; the
investigation closed with a clean, unifying resolution.

## 1. Multi-view / sequence rotation estimator — does NOT break it
Grid-canonicalize each view (de-rotate by the drift-free grid-fine heading → wall frame, removing the
continuous rotation), window-average to denoise, 4-way cross-correlate → relative quadrant. Result: ~44% ≈
single view. Diagnosis: the per-view signal is not just weak, it is MISLEADING — for same-cell views the
cross-corr produces a CONFIDENT peak (42% margin) at the WRONG quadrant ~68% of the time. Averaging misleading
evidence stays misleading. Open cells are locally ≈4-fold symmetric. (multiview_rot.py)

## 2. Richer sensor, wall-bound landmark TYPES — does NOT break it
The env's raycast carries binned landmark types (LBINS≤12); the old sensor collapsed them to binary presence.
Returning the type per ray: ~43% even at landmark density 0.7. Reason: landmarks are WALL-BOUND, and the
recoverable regime is loopy = nearly wall-free → few landmarks → no rotation cue. The structure-vs-
recoverability tension at its root: open space removes the very surfaces sensing relies on. (rich_sensor.py)

## 3. Free-standing beacons (objects/pillars in open space) — partial; LOCAL beacons cap ~60%
Beacons in open cells, sensed by bearing+type. Local beacons (range ~6-8): cross-corr ~45-58%, learned matcher
~58-64% and FLAT with size (better than wall geometry, which declined). But still below ~90%. Local
constellations are themselves rotationally aliased with few/short-range beacons. (beacon_sensor.py)

## 4. THE RESOLUTION — an allothetic orientation reference (a few DISTINCT DISTANT/mid-range cues)
A few distinct cues at distance (a coarse compass / skyline) recover rotation by matching same-type bearings:
- 3-6 distinct DISTANT beacons (R=∞): **96-98%**, FLAT with size.
- 8 beacons at range 10 (~5 cells): **90-93%**, flat — clears the back-end threshold.
- 5 beacons at range 6: 82-86%.

### END-TO-END DEPLOYABLE METHOD WORKS (no oracle rotation):
Full pipeline = grid-fine (drift-free) chain + same-cell loop closures whose relative quadrant comes from the
beacon constellation (grid-corrected: rel = round((beacon_rot − Δgrid)/90)) + Z4-sync (BP) back-end. Global
heading error vs size, loop=0.9, K=6 distant cues, NO oracle rotation:
  17px **1.8°** (q-acc 100%) | 33px **3.1°** (98%) | 49px **3.6°** (98%)  — FLAT, size-invariant.
Two engineering subtleties were essential and now fixed: (a) GRID-CORRECT the closure quadrant (the beacon
gives full relative heading; the back-end's q is grid-fine-relative); (b) closures must be WELL-DISTRIBUTED
across the trajectory (connect sampled steps to their cell's first visit), else BP doesn't converge. At ≥65px
BP needs more iterations (q-acc drops at fixed iters but closAcc stays 100% → pure convergence, iters~size;
[scaling run confirms]). So the recoverable phase is REALIZED end-to-end by an actual estimator with a
realistic allothetic sensor.

## The unifying principle (the real finding)
Orientation in OPEN (recoverable) environments requires an ALLOTHETIC reference; local geometry is insufficient
at EVERY scale:
- LOCAL (closure rotation): open cells are ≈4-fold symmetric → local sensing caps ~60%.
- GLOBAL (the 2-bit frame offset): the maze is ≈4-fold symmetric → unobservable from local views (b08).
ONE mechanism resolves BOTH: an allothetic orientation cue (distant landmark / sun / skyline / compass). b09
showed it makes absolute heading size-invariant (flat to 259px) and collapses the global 2-bit; here we show the
SAME cue supplies the local closure-rotation (90-98%). This is the geometric-module / Cheng-1986 law generalized
to BOTH scales: geometry alone cannot orient; a featural/allothetic cue can — and real open environments have one.

## Net deployable status (positive)
- WITHOUT any allothetic orientation cue: open/recoverable environments are orientation-ambiguous at both the
  local (closure) and global (2-bit) scale → global heading not recoverable from local geometry. (Thoroughly
  confirmed: multi-view, wall-types, local beacons all cap ~45-64%.)
- WITH a realistic allothetic cue (a few distinct distant/mid-range beacons): rotation 90-98% flat → back-end
  recovers (100% given correct closures) → end-to-end size-invariant global heading, no oracle. The deployable
  method = grid-fine (drift-free local) + allothetic orientation reference + Z4-sync back-end (for position).
- Navigation does not need global heading at all (topological), so the practical agent works regardless.
