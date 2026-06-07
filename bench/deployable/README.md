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
- Result (median of 10 OOD mazes, loop=0.9, train n=6,12; `compare.log`):

  | size | cmd-int | GRU(raw) | GRU(+grid) | Transformer | OURS-online |
  |------|--------:|---------:|-----------:|------------:|------------:|
  | 13px | 60.0 | 21.1 | 14.1 | 57.8 | **2.6** |
  | 41px | 41.1 | 29.7 | 25.6 | 47.2 | **1.9** |
  | 89px | 44.2 | 31.8 | 12.5 | 43.7 | **1.8** |

  TRAIN-set error is LOW for all (GRU 7.3, GRU+grid 6.8, Transf 6.6 deg) — so the baselines are NOT
  undertrained; they fit in-distribution and then DRIFT on OOD size (the integrate-and-drift signature). Raw
  GRU/Transformer never discover the drift-free grid-anchoring from raw rays. Feeding the freq-4 grid phase as
  a feature (GRU+grid) cuts error markedly → the win is the INDUCTIVE BIAS, not hand-coding. OURS-online
  (explicit grid-anchoring + recurrent quadrant tracker) is FLAT at ~2deg = size-invariant. (Median; a few
  mazes slip a quadrant without loop closures → mean is higher; closures in the recoverable phase remove them.)

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
