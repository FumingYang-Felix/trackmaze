# Items (b) learned place-rec and (a) iterative SLAM — findings

Goal: turn the recoverable phase (phase diagram) into a working deployable method with NO oracle. Result: the
deployable bottleneck is now precisely localized by clean diagnostics. The theory (recoverable in principle) is
confirmed; the engineering gap is a specific, structural wall.

## (b) Learned place embedding — does NOT break the aliasing wall
Rotation-invariant contrastive embedding (cross-power-spectrum features `[|G|,|L|,|G·conj(L)|]`), trained on
small loopy mazes, eval OOD. Top-1 loop-closure precision: hand-crafted 22-33%, **learned 20-40% (≈ same)**,
learned + oracle-position-gate 33-47%. Learning adds ~nothing over hand-crafted; appearance is intrinsically
aliased in this maze (confirms R4-C). The position gate is the only real lever (R6-C).

## (a) Iterative motion-gated SLAM — bottleneck localized by ablation (the value here)
Pipeline: grid-fine heading (drift-free) + 99.7% per-step chain quadrant → init; propose loop closures
(embedding kNN gated by position) → matcher relative quadrant → robust Z4 back-end → re-integrate position →
tighten gate. Clean diagnostics on n=8..24 loopy:

1. **Per-step chain quadrant: 99.7%** — the backbone is excellent.
2. **Back-end (Z4 BP synchronization): recovers q at ~100% given CORRECT, well-distributed closures**, but
   needs closure accuracy **≳ 0.9** (p=1.0→99%, p=0.85→marginal, ≤0.75→fails) and is fragile to closure
   distribution (concentrated closures fail; trajectory-spread closures work).
3. **Position gate works**: oracle position (radius < 1 cell) → 100% same-cell candidates. The bootstrap
   (estimated position) is chicken-and-egg (the long coverage walk drifts the online front-end → bad gate).
4. **THE BINDING WALL = the relative-quadrant MATCHER on the loopy distribution.** The b03c matcher was 95% on
   DFS data but **33% in-pipeline / 55-65% on clean same-cell pairs on loopy random-walk data**, DECLINING with
   size, and **NOT improved by landmark density** (0.15→0.7 all ~55-65%). 55-65% ≪ the back-end's ~90% need.

## Root cause — a structural TENSION (a genuine finding)
Recoverability of the global frame needs **loopy / 2D-open topology** (phase diagram). But estimating the
**relative rotation** between two views of a place needs **wall geometric structure**. Open/loopy mazes are
nearly wall-free → weak rotation cues → the matcher caps at 55-65%. **The regime where the global frame is
recoverable in principle is exactly the regime where the local rotation cue is weakest.** Denser binary
landmarks don't fix it (uninformative encoding); the impoverished sensor (wall-distance + binary landmark
presence) lacks rotation features that real open environments (textures/objects) would provide.

## Net
- Theory (phase diagram + b15 floor): recoverable in principle in the loopy phase. CONFIRMED, and the back-end
  achieves it given good closures.
- Deployable gap: the relative-rotation matcher in the recoverable (open) regime (55-65% vs ~90% needed). This
  is the precise, structural remaining wall — not place-id, not the solver, not the gate.
- Honest deployable status: the closure-based GLOBAL recovery does not yet work end-to-end in the loopy regime.
  The strong, working deliverables are (1) the phase diagram, (2) item-1 grid-anchoring beating end-to-end
  baselines on OOD global heading, (3) this precise bottleneck map. Navigation does not need global recovery
  (topological), so the practical agent is unaffected.

## Paths forward (for the matcher wall)
- A sequence/multi-view rotation estimator (aggregate rotation evidence over a short window).
- A richer sensor (continuous landmark features / textures) — likely resolves it but changes the benchmark.
- A back-end that tolerates lower closure accuracy (SDP synchronization; exploit the 99.7% chain harder; or
  joint place-rec+sync). Closing the 55-65% → 90% gap is the crux.
