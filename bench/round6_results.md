# Round 6: heading, robust closure, and the breakthrough — MOTION-CONSTRAINED matching beats aliasing

R4–5 localized the open core: vocabulary-independent place recognition under LOCAL ALIASING (oracle local
submap tops out ~0.65 top-1 precision at scale; learned descriptor gives ~0.15 candidate precision in a
metric gate). Round 6 attacks it three ways; the third works.

## 6-A — observation-based heading correction from the Manhattan grid
The freq-4 Fourier phase of the omni view estimates heading **mod 90°** at ~20° circular std, **perfectly
size-invariant** (zero bias). But the **4-fold quadrant** is irreducible from locally-symmetric walls; the
only quadrant cue is command integration, which random-walks (even at rot_noise=0.02 the fused absolute
heading is 57–111° RMS, quadrant-correct ~0.2–0.6 at large sizes). → heading mod 90° is boundable from
observation; ABSOLUTE heading needs a persistent global cue = a reliable loop closure. Heading and loop
closure are the same problem.

## 6-B — ROBUST loop closure (Dynamic Covariance Scaling) over learned candidates
DCS should down-weight closures inconsistent with the trajectory. Result: **w_true = w_false = 1.00** — it
rejects nothing. Robust estimators have a <50% breakdown point; here inliers are ~10–17% (the MAJORITY are
false), so the optimizer just fits the false closures (small residuals) and keeps them. Stricter matching
(mutual-NN + Lowe ratio + tight gate + high threshold) cuts candidate COUNT 1070→222 but leaves precision
~0.10–0.16 unchanged: **aliased different-cells have descriptors as similar as true revisits — appearance
genuinely cannot separate them, confirming R4-C from every angle.**

## 6-C — the breakthrough: MOTION-CONSTRAINED topological matching
The fix is to stop matching each observation against ALL places (a loose ~50-candidate metric gate where
appearance is hopeless) and match only against the few places reachable in ONE step from where you just were
— the graph-neighbors of the current topological node (~4 candidates). The maze's transition structure does
the disambiguation appearance can't. Online topological mapper, rotation-invariant descriptor trained on
n=6,12, evaluated OOD:

| n (grid)   | MOTION-CONSTRAINED purity / node-per-cell | GLOBAL-appearance purity / node-per-cell |
|------------|-------------------------------------------|------------------------------------------|
| 6 (13×13)  | **0.81** / 1.53                           | 0.28 / 0.35                              |
| 20 (41×41) | **0.81** / 1.55                           | 0.07 / 0.08                              |
| 40 (81×81) | **0.83** / 1.58                           | 0.025 / 0.03                             |

**Motion-constrained: node purity ~0.83, perfectly size-invariant, ratio ~1.55 (benign over-split — some
duplicate nodes from missed different-route revisits, but NO catastrophic false merges).** Global appearance:
purity collapses 0.28→0.025 and ratio →0.03 (97% of cells wrongly merged into shared nodes — a corrupt map).

## The synthesis (the whole R3→R6 arc)
- **R3:** memory is NOT the wall — O(1) importance-weighted memory matches unbounded; residual = √ single-
  gauge-anchor floor (fundamental). (Refuted my own "pigeonhole" over-claim.)
- **R4–5:** local APPEARANCE cannot do high-precision global place recognition at scale (aliasing; oracle
  ceiling 0.65; robust SLAM breaks at minority-inlier). Place-recognition AUC itself is size-invariant though.
- **R6:** heading mod 90° is observation-boundable but the quadrant needs a loop closure (6-A); robust back-
  ends can't rescue minority-inlier candidates (6-B); **the disambiguation must come from the MAZE's
  TRANSITION STRUCTURE, not appearance — match against topological neighbors, not all places (6-C).**

The size-invariant ingredients now in hand: LOCAL tracking (R2), O(1) memory (R3), a size-invariant learned
place descriptor (R4), and **motion-constrained topological matching (R6-C) that is robust to aliasing and
size-invariant.** Together they build a locally-consistent, navigable topological map of an arbitrarily large
maze, trained on small ones.

## Next (Round 7): from map to NAVIGATION (the user's actual goal)
The decomposition is now clean: **completeness** (finding the exit) is size-invariant for free via local
exploration; **optimality** (near-shortest-path) needs the map — and 6-C gives a size-invariant one. Build
the full agent: explore with the motion-constrained topological mapper, recognize the goal, plan on the
graph (return / shortest-path), and report SPL (success-weighted path length) vs size vs an oracle
shortest-path and vs the memorizing baselines. Refinement (merge duplicate nodes via occasional verified
global closures; the user's grid-code idea-B can serve as the drift-bounded global phase that proposes safe
merges) is optional polish, not a blocker. The thesis: every component is size-invariant by construction, so
the agent trained on 13×13 navigates 105×105 — which the end-to-end memorizing nets (phase-1) cannot.
