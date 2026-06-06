# Round 11 (A — navigation): the reframed goal is solved size-invariantly, but NOT by the learned tracker

Goal (user's reframe): reach the exit WITHOUT getting lost (no redundant re-walking), size-invariantly; NOT
shortest path. Test: DFS explore-to-exit with loop detection from an online position estimate (perfect / raw
odometry / the trained MGM run online). Low-level execution oracle (like R7-C). Metric: success + steps
relative to the perfect-map DFS (ratio = redundancy; lower = less "getting lost"). ckpt = mgm_v2_big.

## Results (6 mazes/size; ratio = steps / perfect-DFS steps)
At a CONSERVATIVE loop-detection radius (r=0.4):
| n (grid)   | perfect | odo (succ/ratio) | mgm (succ/ratio) |
|------------|---------|------------------|------------------|
| 12 (25x25) | 1.0/1.0 | 1.0 / 1.6        | 1.0 / 1.0        |
| 20 (41x41) | 1.0/1.0 | 1.0 / 2.4        | 1.0 / 2.2        |
| 28 (57x57) | 1.0/1.0 | 1.0 / 0.9        | 0.83 / 1.8       |
(r=1.2 loose: mgm collapses to 0% — false-positive loops skip the exit's branch; r=0.8 mgm 67-83%.)

## The honest findings
1. **The reframed goal is achievable and size-invariant.** With conservative (no-false-positive) loop
   detection, the agent reaches the exit with bounded redundancy (ratio ~1-2.5, i.e. within 1-2.5x of optimal
   exploration) from 13x13 to 57x57. "Find the exit without getting lost" generalizes in size.
2. **The learned MGM does NOT beat simple odometry for this.** odo is as good or MORE robust (100% at all
   sizes; mgm occasionally drops to 83%). The navigation-completeness bottleneck is CONSERVATIVE loop
   detection (avoiding false merges that skip branches), and estimation ACCURACY does not help it -- worse,
   the MGM's loop-closure *re-anchoring* actively CREATES false positives (at loose r it fails completely),
   while odometry drift is benignly conservative (different places stay separated in estimate space).
3. So the two stages do NOT compose the way hoped: an accurate re-anchoring tracker is *anti-conservative*,
   which is exactly wrong for DFS-completeness. (Sharpens the R7-C lesson.)

## Net project verdict
- **Stage-1 (state tracking): a real learned-arch win.** mgm_v2_big = size-invariant LOCAL accuracy (~0.09 to
  129x129, ~10x training), beats GRU/LSTM/Transformer OOD, clean ablations (motion-gated memory is the
  driver). Global metric drift is the fundamental sqrt gauge floor (R3), slowed but not removed.
- **Stage-2 (navigation, find-exit-without-getting-lost): solved size-invariantly by a SIMPLE conservative
  DFS + odometry loop-detection -- the learned tracker is not needed for it.** The accurate tracker's value
  is state estimation, not navigation-completeness.
- **Near-optimal (shortest-path) navigation remains open** (needs the aliasing-robust loop closure = the
  R4-C wall) -- but the user reframed away from requiring it.

## Where the MGM could still pay off in navigation (untested, honest leads)
- HOMING / return trips: after exploring, navigate back to start/exit by the estimated map -- needs accurate
  global state (odo's drift makes the return blind; MGM should win here). Not yet tested.
- HIGH-drift regimes where odometry drift breaks even conservative loop detection -- the MGM's loop-closure
  could rescue it. Current env (rot_noise=0.09) doesn't stress this enough.
- A learned navigation POLICY (RL) on top of the MGM state, end-to-end, rather than a hand-coded DFS.
