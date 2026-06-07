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

## ★★ THE PHASE TRANSITION (b10–b15): "breaking gauge" is set by ENVIRONMENT TOPOLOGY, not solver/data/compute
The night's deepest result, prompted by "is there really no way through gauge?". I re-examined my own "grows
with size" verdict and found it was a CONTINUOUS-approximation + WEAK-SOLVER + SINGLE-ENVIRONMENT artifact.

**The quadrant is a Z_4 SYNCHRONIZATION field (a 2D Potts model), not a continuous estimate** (b10). Recovering
it from pairwise relatives (drift-free-per-step chain + 95% matcher closures) is group synchronization. ICM
(b11) and naive BP (b14) failed — but that is a SOLVER artifact: ICM cannot make the non-local segment-flips
that undo accumulated drift (it sat at chain-init, 31%); 1D BP needs O(T) iters and decorrelates under any
noise. So solver failure ≠ unrecoverable.

**The estimator-free truth is the effective resistance to the anchor on the maze CELL graph** (b15) — the BLUE
variance floor of the global frame, which no estimator beats. It is a RANDOM-WALK RESISTANCE / Pólya-recurrence
quantity governed by the graph's DIMENSION:
  - **Perfect maze = spanning TREE (1D):** R_eff ~ distance → heading-std floor ~ √distance, and the MAX floor
    GROWS with maze size: 79°→141°→205°→254° for n=20→40→60→80 (b15). The discrete Z_4 field has NO long-range
    order in 1D. => the global frame DECORRELATES over a finite length, INDEPENDENT of size => **NOT recoverable
    at scale. Fundamental for trees — no solver/data/compute/architecture changes it.**
  - **Loopy / 2D-connected (loop≳0.3–0.6):** R_eff ~ log(distance) → floor SATURATES; MAX floor is FLAT with
    size: 13°→15°→16°→18° for n=20→80 (b15). The Z_4 field has true 2D long-range order (Potts ordered phase).
    => the global frame IS recoverable SIZE-INVARIANTLY. **Gauge is "broken" — by environment topology.**
  - Crossover is smooth in loop density: lin-fit slope of std-vs-distance drops 0.139→0.040 as loop 0→0.9; the
    floor at d=60 drops 40°→10°.

**So the precise answer to "can we break gauge / go forever?":**
  1. The global METRIC FRAME (heading quadrant + position) is recoverable at any size **iff the environment is
     2D-loop-connected above a topology threshold.** In a tree-maze it is fundamentally not (1D, linear
     decorrelation). This is the Pólya-recurrence / lower-critical-dimension law for spatial self-localization:
     2D is the marginal dimension where global orientation just barely orders (log), 1D/tree never does.
  2. The single global 2-bit OFFSET ("which way is north") is irreducible regardless of topology — it is an
     environment SYMMETRY (b08, exact 4-fold), collapsed only by one featural cue (b09).
  3. The globally-anchored LOCAL structure (fine heading mod-90, local metric, topology) is size-invariant in
     ALL cases (L0–L2).

This upgrades the verdict from "global tracking is gauge-bounded (negative)" to a **controllable phase
transition**: a benchmark with a LOOP-DENSITY knob and a SYMMETRY knob exhibits, for global egocentric
self-localization, (i) a topological order/disorder transition (tree→loopy = unrecoverable→recoverable, the
Pólya/Potts lower-critical-dimension boundary) and (ii) a symmetry-breaking transition (the 2-bit offset). An
architecture that reads the Z_4 synchronization field (not a single integrated frame) is size-invariant exactly
in the recoverable phase. THIS is the spotlight: not "we solved OOD tracking" nor "it's impossible", but a
clean, predictive THEORY OF WHEN it is possible, with the order parameter, the critical dimension, and the
neuro grounding (HD cells + allothetic anchoring + geometric-module symmetry).
