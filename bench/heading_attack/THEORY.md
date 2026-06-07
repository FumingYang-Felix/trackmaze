# The gauge structure of OOD-size egocentric state tracking — what is recoverable, and what is irreducible

This is the theoretical spine behind the b01–b11 experiments. The question the experiments answer:
**which components of egocentric state can be tracked size-invariantly (train small → test arbitrarily large),
and which are fundamentally gauge-bound?** The answer is a clean hierarchy, each layer with a different limit.

## Setup
Agent has only egocentric, relative sensing: an omni view (raycast distances `g` + landmark ids `l`) and its
own commanded turn/step. Persistent heading random-walk `rot_noise=σ` per step; odometry noise. The maze walls
are axis-aligned (a global 4-fold-symmetric orientation field). Heading decomposes as
  h_t = φ_t + (π/2)·q_t ,   φ_t ∈ [0,π/2) "fine",  q_t ∈ Z_4 "quadrant".

## The hierarchy (from fully recoverable to irreducible)

### L0. Fine heading φ_t (mod 90°): GLOBAL-LANDMARK anchored → drift-free, size-invariant. ✓
The walls are visible everywhere, so the freq-4 Fourier phase of the omni view reads φ_t directly, every step,
**without integration**. Measured: ~5° noise, FLAT 13×13→259×259 (b06). This is an *allothetic* measurement,
not path integration, so it has no drift and no size dependence. (Earlier "20°" was a sign/reflection artifact;
`_grid_heading` is reflected — negate it. True noise ~5°.)

### L1. Local windowed metric (displacement over a short window): size-invariant. ✓
Relative displacement over W steps depends only on W noisy increments, not on path length → variance is O(W),
independent of maze size. This is "navigable accuracy". (Established in earlier rounds.)

### L2. Topology (place graph): size-invariant. ✓
Adjacency/visit structure is combinatorial; no metric drift. Navigation ("find exit, don't re-walk") lives here.

### L3. Quadrant FIELD q_t (the relative quadrants among all visits): Z_4 SYNCHRONIZATION.
This is the crux and where my earlier verdict was wrong. Recovering q_t from pairwise relative measurements
(chain: `round((turn−Δg)/(π/2))`, ~drift-free per step; closures: the 95% learned matcher, b03c) is **group
(angular) synchronization over Z_4** = a **2D Potts model**, NOT a continuous estimation problem.
  - The CONTINUOUS analogue (b05 effective resistance) is a 2D **Gaussian free field** → variance ~ log(size)
    or √ along a tree → genuinely unbounded. I wrongly applied this to the discrete field.
  - The DISCRETE field has a **threshold**: below a critical edge-noise / above a critical closure connectivity,
    it has **long-range order** → bounded fluctuation → recovery up to a single global offset, **independent of
    size**. Above the noise threshold (or tree-like sparse graph) → disordered → errors accumulate.
  - Therefore recoverability is a **coverage/closure-density trade-off**, not a hard size wall:
      • DFS explore-once → graph ≈ a path+sparse closures (tree-like) → no rigidity → q drifts (matches b05).
      • Dense revisiting → graph → 2D-lattice-like → Potts order → q recoverable size-invariantly (the cost is
        O(extra traversal) for loop closures — exactly what SLAM trades coverage for).
  - Solver matters: spectral sync fails on path-like graphs (b10); a discrete solver using cycle redundancy
    (max-product / ICM / BP, b11) is required.
  [b11 fills in: is ORACLE-dense FLAT with size? → discrete field IS size-invariantly recoverable given density.]

### L4. Global metric position (x_t, y_t relative to start): continuous 2D gauge field. ✗ (but unneeded)
Even with dense loop closures this is a 2D Gaussian free field referenced to a single anchor → variance grows
(≥ logarithmically) with distance-from-start. Genuinely gauge-bound. BUT navigation does not need it (L2/L1
suffice), and "global coordinates" is the wrong objective (no animal maintains them either).

### L5. The global 2-bit offset q_0 (which wall direction is "north"): IRREDUCIBLE. ✗ (and that's fine)
The maze is (near-)exactly 4-fold rotationally symmetric in local appearance → the absolute orientation of the
whole frame is **information-theoretically unobservable** from any amount of local egocentric data. PROVEN: a
learned classifier reads absolute quadrant from one view at **balanced OOD = chance (25%)**, declining with size,
despite 95.8% train (pure memorization) (b08). No data/compute/architecture can break this — it is a SYMMETRY of
the environment, not a deficiency of the method. ONE omnipresent featural cue (compass/sun/direction-colored
walls), even 10%-noisy, collapses it → full absolute heading size-invariant (b09, FLAT 13×13→259×259).

## Two classical groundings (this is not ad hoc)
- **Gauge theory / lattice field theory**: observables defined w.r.t. a GLOBAL field (the walls) are bounded;
  observables defined w.r.t. a SINGLE point (the start) integrate noise. Discrete gauge field (Z_4 Potts) orders
  in 2D below threshold; continuous (Gaussian) gauge field does not (Mermin–Wagner-flavored: the discrete vs
  continuous symmetry distinction is exactly why L3 can order while L4 cannot).
- **Neuroscience**: head-direction cells path-integrate (drift, the L3/L4 problem) and re-anchor to allothetic
  cues (the L0 walls); the irreducible residue L5 is precisely the **geometric-module rotational error** (Cheng
  1986): animals in a 2-fold-symmetric rectangular arena make systematic 180° place errors and **cannot break
  the symmetry with geometry alone** — they need a featural cue (a colored wall). Our 4-fold maze is the same law.

## The spotlight claim
OOD-size egocentric state tracking is NOT one problem with one limit; it factors into a **size-invariant,
globally-anchored part (L0–L3, recoverable to the information floor at ANY size, the discrete pieces via Z_4
synchronization above a coverage threshold)** plus a **gauge residue (L4 continuous position + L5 the 2-bit
symmetry offset) that is provably irreducible but functionally unnecessary**. Standard RNN/Transformer baselines
fail because they try to integrate ONE global frame (drifting on L3/L4); an architecture that locks the
recoverable factorization (grid-anchored φ + topological place graph + local metric + a synchronization read-out
for q) is size-invariant **by construction**, and treats L5 as a settable latent (a featural cue when present).
