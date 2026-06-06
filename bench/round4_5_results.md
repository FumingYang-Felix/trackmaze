# Rounds 4–5: the real wall is vocabulary-independent place recognition under LOCAL ALIASING

R3 refuted the "memory" framing (O(1) compressed memory matches unbounded; residual = √ gauge floor). It
redirected to the genuinely open part R3 had cheated with oracle place-identity and R2-B failed at: LEARNED,
aliasing-robust LOOP-CLOSURE RECOGNITION. R4–5 attack that directly.

## R4-A — is place-recognition discriminability size-invariant? (raw, no training)
Wide-coverage (oracle DFS) data so large mazes actually have thousands of distinct, potentially-aliased
cells. AUC(same-cell vs different-cell) of the allo-canonical omni view: **~0.85, FLAT from 13×13 to
105×105** (lm_hist 0.74, omni 0.82→0.86, omni+ctx ~0.84). → discriminability does NOT collapse with size;
the recognition task is size-invariant. Good news for OOD — but 0.85 AUC is far too low for loop closure.

## R4-B — learned contrastive descriptor, trained small, eval OOD
A contrastive encoder (MLP, and a rotation-INVARIANT circular-conv = heading-free) trained on n=6,12 only:
AUC ~0.91 on 13×13 and **holds ~0.90–0.92 on unseen 41×41…105×105** (transfers OOD, size-invariant). BUT
precision@80%-recall with 5× negatives is only **~0.5** → too many false matches for loop closure.

## R5 — capstone: learned + metric-gated loop closure end-to-end (oracle-heading odometry regime)
Rotation-invariant descriptor proposes matches; a metric gate (estimate within radius R) filters; accepted
pairs become R3-B pose-graph loop edges. Result: **loop-closure precision ~0.17** → false matches corrupt
the graph → learned (43.0 at n40) is an order of magnitude WORSE than plain dead-reckon (3.0). Why: a gate
of radius R contains ~πR² candidate cells; AUC 0.91 over k candidates ≈ 0.91^k → collapses. Loop closure
needs ≈0.99 AUC, not 0.91. (Also exposed: in the low-drift oracle-heading regime, the loop-closure
constraint "same cell ⇒ same position" has a ~1-cell sub-cell error floor, so it can even hurt — part of why
R3-B's compressed graph "beat" full.)

## R4-C — the ceiling: ORACLE local-submap descriptor (true local geometry)
Is the descriptor just under-trained, or is the environment fundamentally aliased? Test the CEILING: the
true (2W+1)² wall+landmark patch around each cell. Gated top-1 retrieval precision (R=4):

| n (grid)   | W2 AUC/top1 | W4 AUC/top1 |
|------------|-------------|-------------|
| 6 (13×13)  | 0.93 / 0.69 | 0.99 / 0.83 |
| 20 (41×41) | 0.90 / 0.66 | 0.95 / 0.67 |
| 40 (81×81) | 0.89 / 0.65 | 0.92 / **0.65** |

Even the **oracle** local submap tops out at ~0.65 top-1 precision at scale, and DROPS with size. Unique
landmarks (ambiguity=0) + denser (0.35) barely change it (~0.66). → **the maze is locally self-similar:
different cells have identical local geometry, so NO appearance descriptor — learned or oracle — can place
them by appearance alone, and it gets worse as size grows (more locally-identical cells per metric gate).**

## The synthesis (the honest wall)
The size-OOD of state tracking is NOT memory-limited (R3) and NOT discriminability-limited in the size sense
(R4: AUC flat). It is limited by a **two-horned vocabulary-independent recognition problem**:
- match landmarks by **id** ⇒ trivial but does NOT generalize OOD (large mazes have unseen ids — the phase-1
  "landmarks HURT" memorization trap);
- match by **local geometry** ⇒ aliases (oracle ceiling ~0.65 top-1 at scale).
Either way, **local appearance cannot reliably identify a place at scale.** And the metric gate that should
rescue precision loosens as global drift grows — and global drift is dominated by HEADING (R3 cmd-heading
p≈1.2), creating a vicious cycle: heading drift → large position drift → loose gate → bad loop-closure
precision → can't re-anchor heading → …

## What breaks the cycle (the next directions — both re-motivated from first principles, not arbitrary)
1. **Observation-based heading correction from the maze's Manhattan grid.** Axis-aligned walls give an
   absolute heading reference mod 90° in the omni view (a frequency-4 component). Correcting heading to the
   grid every step bounds the heading random-walk → bounds position drift → keeps the gate tight. This is the
   per-step mechanism behind R2's flat LOCAL, made explicit and absolute (not command-integrated).
2. **Path-integrated disambiguation of aliased places — the user's idea B (grid/toroidal code), now with a
   concrete job.** Two identical-looking junctions are told apart not by appearance but by the continuous
   path-integrated phase that leads to each. A modular grid code disambiguates aliasing and extends to large
   sizes (range = LCM of module periods ≫ any single period). This is exactly why grid cells exist. Round 6
   target: show a path-integrated (grid/attractor) position prior disambiguates aliased loop closures where
   appearance can't, size-invariantly — turning ~0.65 appearance-precision into reliable closure.

So: memory — solved; size-invariant local tracking + descriptor — solved; the open core is **binding aliased
local appearance to a drift-bounded, size-extensible path-integrated global code.** That is the method to build.
