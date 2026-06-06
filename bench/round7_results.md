# Round 7: from map to navigation — completeness is size-invariant, OPTIMALITY hits the loop-closure wall

The thesis: every component is size-invariant by construction, so a small-trained agent navigates large mazes.
Round 7 tests whether the R6-C topological map turns into navigation.

## The metric matters: finding an UNKNOWN goal is exploration-dominated
First cut measured SPL to an unknown goal. But that SPL → 0 with size for ANY method including a PERFECT map
(verified: perfect-map DFS reaches the goal at all sizes but visits ~O(area) cells to find a hidden goal,
while the shortest path is ~O(n), so SPL ~ 1/n by necessity). So SPL-to-unknown-goal is the wrong thesis
metric — it measures the unavoidable cost of searching for a hidden target, not the map's quality.

## The right metric: is the learned map NAVIGATIONALLY FAITHFUL? (7-B)
Build the online map, then compare shortest-path distance ON THE LEARNED GRAPH to the TRUE shortest path for
many (start,goal) pairs. Ratio ~1 & flat ⇒ near-optimal navigation possible at any size.

| n (grid)   | MOTION-constrained ratio / reach / node-per-cell | GLOBAL-appearance ratio / reach |
|------------|--------------------------------------------------|---------------------------------|
| 6 (13×13)  | 5.1 / 0.97 / 2.9                                  | 0.41 / 0.98                     |
| 20 (41×41) | 22.2 / 1.00 / 3.0                                 | 0.12 / 1.00                     |
| 40 (81×81) | 50.8 / 1.00 / 3.1                                 | 0.06 / 1.00                     |

- **MOTION-constrained map is a TREE.** Motion-constraint links only consecutive traversals and refuses
  different-route revisits (to avoid false merges) — so it never recovers the maze's LOOP-CLOSURE edges (the
  braided shortcuts). Result: it's connected and reaches everywhere (reach ~1.0 → **COMPLETENESS is
  size-invariant**), but graph paths are tree-paths, 5–50× longer than true shortest paths, and the ratio
  GROWS with size (distant tree-paths get arbitrarily worse than loop-using shortest paths).
- **GLOBAL-appearance map has shortcuts but is CORRUPT** (ratio < 1 = false "teleports" from merging
  different cells that look alike; the R4-5 aliasing wall).
- Local k-hop loop closure (match within k graph-hops) helps partially (k=3: ratio 2.4→25 across sizes,
  purity 0.73) but the ratio still grows with size — local appearance can't reliably confirm even local loops.

## The honest verdict (the whole session, R3→R7)
What is solved and **size-invariant**: LOCAL tracking (R2), O(1) memory (R3, my pigeonhole over-claim
refuted), a learned place descriptor (R4), motion-constrained mapping with NO catastrophic merges (R6-C), and
therefore **navigation COMPLETENESS** — a small-trained agent builds a connected, navigable map of an
arbitrarily large maze and can reach any goal. That alone beats the phase-1 memorizing baselines (which don't
transfer at all).

What remains genuinely OPEN: **navigation OPTIMALITY (near-shortest-path) requires recovering the maze's
loops = reliable loop closure under LOCAL APPEARANCE ALIASING.** Every angle hits the same wall: appearance
caps at ~0.65 oracle precision (R4-C), robust SLAM breaks at minority-inlier (R6-B), motion-constraint avoids
false merges only by also dropping true loops (7-B). This is not a memory or a discriminability limit; it is
the vocabulary-independent data-association problem, and it is the real, hard, still-unsolved core.

## Candidate attacks on the open core (for the next push — none yet proven)
1. **Geometric loop closure on SHORT loops.** A braided cycle is short; over a short recent window odometry
   drift is small, so "you've returned to a node ~1 cell away" is geometrically verifiable WITHOUT appearance.
   Adds the local shortcuts the tree misses, without false merges. (k-hop tried appearance; this uses geometry.)
2. **Active sequence verification.** Confirm a proposed closure by checking the agent's NEXT few observations
   match the map's prediction along that path — a sequence is far more unique than one view; a false closure
   diverges fast.
3. **The user's idea B — a modular grid/toroidal code** as the drift-bounded, size-extensible global phase
   that proposes safe closures (disambiguates look-alike junctions by their path-integrated phase). `toy_torus`
   showed bounded error on 10× OOD; the open step is multi-module disambiguation grafted onto the maze.

The thesis stands for completeness; optimality is the honest frontier. Not defeatist (every piece IS
size-invariant; the gap is sharply localized to one sub-problem with concrete candidate attacks) and not
overclaiming (near-optimal size-OOD navigation is NOT yet demonstrated).
