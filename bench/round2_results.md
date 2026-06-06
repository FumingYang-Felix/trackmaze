# Rounds 1–2: the size-OOD of state-tracking, split

Goal: train on 13×13, generalize ~infinitely in maze size.

## The derivation
A local, size-invariant recurrent update keeps error bounded at any size **iff** its error dynamics are
contractive (|A|<1). A vanilla integrator has |A|=1 → error ~ √(path) → grows. Contraction must come from
**observation-driven correction at a constant rate** (constant landmark density ⇒ constant re-anchor rate
⇒ error bounded independent of size).

## The rounds
- **R1 — explicit contraction loss (perturbation-recovery): FAILED.** Random-perturbation contraction does
  not provide observation anchoring; it can't correct *unobservable* drift. err-vs-size grew (+3.5) for all λ.
- **R2-A — oracle re-anchor: CONFIRMS the mechanism.** Re-anchor whenever a landmark is in view → steady-
  state error ~1, FLAT from 13×13 to 121×121 (0.54→0.36). Constant-rate local re-anchor bounds error at any size.
- **R2-B — learned local re-anchor: HURTS, but reveals the key split.** Online proximity-association re-anchor
  destabilizes (wrong associations blow up). BUT the integrator's **LOCAL** error (relative displacement over a
  30-step window) is already flat. Continuous use of the omni observation gives size-invariant LOCAL accuracy.
- **R2-C — confirmed (3 seeds, to 105×105):** LOCAL error FLAT/decreasing (0.95→0.37, grow −0.55);
  GLOBAL error GROWS (3.2→ ~5–8, grow +2.0).

## The honest finding (the split)
The size-OOD of state-tracking SPLITS:
- **LOCAL state (navigable accuracy) — size-invariant, ACHIEVED.** A learned integrator that uses the local
  observation every step has flat LOCAL error from 13×13 to 105×105. This is the part navigation needs
  (move correctly relative to your surroundings).
- **GLOBAL metric position — grows; the genuinely hard part.** Displacement-from-start drifts; bounding it at
  any size needs either loop-closure (whose rate DROPS with size → still grows) or a modular/grid code (range
  extends but emergence is fragile, topology must be pre-matched — MapFormer). With fixed memory this is
  fundamentally limited (a fixed code can't keep an unbounded space globally consistent).

**So "near-infinite size generalization" is achieved for the LOCAL/navigable state, and is fundamentally
memory-limited for global metric.** For the navigation TASK (find exit), local accuracy + a topological map
(which grows with explored area — unavoidable) is what's needed; per-step computation is size-invariant.
