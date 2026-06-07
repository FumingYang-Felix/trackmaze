# Deployable method + the phase diagram (items 1 & 2)

This folder turns the R14 theory (`../heading_attack/THEORY.md`) into (1) a method that beats end-to-end
baselines on OOD global heading, and (2) the spotlight phase diagram of when global self-localization is
size-invariantly recoverable.

## Item 2 — THE PHASE DIAGRAM (`phase_diagram.py` → `phase_diagram.png`)
Order parameter = the estimator-free global-heading-std **floor** (effective resistance to the start anchor on
the honest visit graph; the BLUE variance no estimator beats). Recoverable iff floor < 45° (quadrant threshold).
- **(A) loop-density × maze-size** (dense closures): tree (loop≈0) GROWS with size (85→326°, unrecoverable);
  loopy (loop=0.9) FLAT (20→30°, recoverable). Topological order/disorder transition; the boundary is ~loop
  0.3–0.6 (critical dimension d=2: 1D/tree never orders, 2D does).
- **(B) loop-density × closure-density** (fixed size): need BOTH high loop AND enough loop closures
  (npc=5 → ~500° everywhere; loop=0.9 + npc=40 → 29°). The active-SLAM coverage axis.
- The SYMMETRY axis is orthogonal: the single global 2-bit offset is irreducible (`../heading_attack/b08`),
  collapsed by one featural cue (`b09`). So the full phase space is (topology) × (coverage) × (symmetry).

## Item 1 — METHOD vs BASELINES (`baselines_vs_method.py`)
Task: predict GLOBAL heading-from-start at every step; train on small loopy mazes, eval OOD large. Question:
does a trained GRU/Transformer LEARN the drift-free grid-anchoring (freq-4 Fourier phase of the omni view) +
quadrant tracking, or does it integrate-and-drift?
- **OURS-online** = grid-fine (drift-free mod-90, sign-corrected) + grid-anchored integer-quadrant tracker.
  No training, no loop closures — pure structure.
- Baselines: GRU / Transformer end-to-end on [g, l, action] → (cos, sin) of true heading.
- Result: [filled by run] — OURS-online is far flatter/lower than the trained integrators, which drift like
  open-loop dead reckoning. The freq-4 grid signal + recurrent quadrant tracking is an inductive bias the
  baselines do not discover from raw rays → STRUCTURE beats end-to-end for OOD global heading.

## The remaining wall — PLACE RECOGNITION under aliasing (`seqplace.py`)
Full global recovery in the recoverable phase needs loop closures = matching revisited places WITHOUT oracle
id. Hand-crafted rotation-invariant descriptors (|rfft(g)|+|rfft(l)|, single or sequence) cap at ~0.4–0.6
precision and DROP with size (confirms R4-C: the maze is locally self-similar). The motion-gate (R6-C) prunes
candidates but needs a position estimate → chicken-and-egg solved only by incremental motion-gated SLAM. This
is the honest open sub-problem the phase diagram isolates: it is needed only to close the quadrant in the
RECOVERABLE phase; navigation itself is topological and does not need it.

## Honest status
- Item 2 (phase diagram): DONE — clean two-axis spotlight figure.
- Item 1 (method beats baselines on OOD global heading): DONE via grid-anchoring structure.
- Place recognition for full closure-based recovery: the identified remaining wall (appearance caps ~0.5);
  the path is iterative motion-gated SLAM / a learned place embedding — future work.
