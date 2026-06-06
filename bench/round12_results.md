# Round 12 (challenge option 2 — fewer steps): a multi-agent loop-closure search + an honest verdict

Goal: near-optimal (fewer-step) size-invariant navigation, which needs reliable LOOP CLOSURE to recover the
maze's shortcuts — the R4-C aliasing/heading-drift wall. Built a faithful CPU substrate (`round12_sim.py`):
frontier nav with a SIMULATED tracker est = rotate(true_disp, theta_drift) + transl_drift, calibrated
(sigma=0.10, sigma_rot=0.04 = MGM_OP) to reproduce the real trained-MGM nav (tight-gate ~100% small ->
~50% at n40). Ran an 11-agent Workflow: 8 strategies implemented+tested in parallel, top-3 adversarially
verified. score() = completeness - wander penalty; TightGate baseline = 0.819 (n40 success 0.50).

## What the search found (with the agents' own honest caveats)
- **Pure metric/est-based loop closure CANNOT beat baseline.** Multiple independent agents tried de-rotation,
  global-frame fixes, per-node-theta correction, multi-hypothesis, strict gating — all ~baseline. Direct
  quotes: "spurious same-shape |est-est_j| distributions FULLY OVERLAP and even a strict est-only gate tops
  out near precision ~0.5"; "de-rotation can't cancel the translational drift random-walk." This RIGOROUSLY
  confirms the heading-drift / R4-C wall: you cannot recover reliable loop closure from the drifting estimate.
- The score-1.0 winners (`relgate`, `wildcard`) work by **drift-free TOPOLOGICAL coordinates**: sum the
  cardinal direction labels stored on the graph's edges along a BFS path -> exact integer relative position,
  zero accumulated rotation at any length -> exact loop closure (holds even at 5x drift, to 113x113). The
  agents FLAGGED this honestly as a SUBSTRATE LEAK: my substrate stored the TRUE world-cardinal direction `d`
  on each edge, which is drift-free; a real agent only knows the direction in its OWN drifting heading frame.
- The honest heading-corrected est version (relgate v1) scored 0.828 — barely above baseline. `posegraph`/
  `robustpose` (exact odometry-edge positions + de-rotated single-step) reached 0.91-0.97 but ALSO lean on
  the true-direction edge labels; their n40 success is 0.67-0.88 (seed-optimistic).

## The honest verdict on "fewer steps" (option 2)
Reliable loop closure (-> fewer steps, near-oracle, size-invariant) IS achievable — but ONLY via drift-free
**topological coordinates built from direction LABELS**, NOT from the metric estimate. The metric estimate
provably can't do it (est-based tops out at ~0.5 precision). And the topological approach is legitimate iff
the agent can correctly LABEL which cardinal direction each corridor traversal was — i.e. heading accuracy
< 45 deg (so the nearest-cardinal label is right). That is exactly the R6-A regime: the Manhattan grid gives
heading mod 90 deg at ~20 deg (size-invariant), but the QUADRANT (which of 4) drifts and flips over long
trajectories at high noise. So:
  - low heading drift (quadrant maintained) -> correct labels -> drift-free topology -> reliable closure ->
    fewer steps (near-oracle), size-invariant.
  - high heading drift (quadrant flips) -> wrong labels -> corrupt topology -> fails.
This unifies the whole project: the binding constraints are GAUGE (R3) and ABSOLUTE HEADING / the quadrant
(R6-A), both fundamental. Fewer-step navigation is gated by absolute heading, not by the metric tracker.

## Status of artifacts
`closer_*.py` (relgate/wildcard/posegraph/robustpose/multihyp/seqverify/...) kept under bench/ (research
files, not deleted). NOTE the relgate/wildcard score-1.0 use the substrate's true-direction edge labels —
treat them as the topological MECHANISM demo, not a heading-drift solution. The rigorous, transferable
finding is: est-based loop closure fails; topological-label closure works iff heading labels are correct.

## Next: option 1 (the user's plan: "先挑战2，然后进行1")
A fully-learned navigation policy. Informed by R11/R12: maze nav is TOPOLOGICAL, so the policy should consume
the topological/relational state + the drift-free commanded-action history, and (per R6-A) a heading-mod-90
cue — NOT rely on the drifting metric MGM state, which R11 showed does not help high-level navigation.
