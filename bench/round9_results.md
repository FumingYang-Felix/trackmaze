# Round 9: extreme-OOD confirmation + idea-B test (honest negative)

Promote mgm_norot (R8 best); push OOD to n=64 (129x129 = ~10x the n=6/12 training); test idea-B (a
multi-scale periodic memory gate: coarse scale tolerates drift, fine scale disambiguates over a large range)
vs the plain Gaussian motion-gate. 4 archs x 3 seeds on FASRC (job 19561717).

## Results (mean+-std over 3 seeds; train = n6,12)
LOCAL error (navigable accuracy):
| n (grid)     | mgm_norot | mgm_grid | mgm_grid_M128 | gru  |
|--------------|-----------|----------|---------------|------|
| 52 (105x105) | 0.11      | 0.11     | 0.13          | 0.27 |
| 64 (129x129) | **0.11**  | **0.11** | 0.14          | 0.31 |

GLOBAL error (final displacement):
| n (grid)     | mgm_norot | mgm_grid | mgm_grid_M128 | gru  |
|--------------|-----------|----------|---------------|------|
| 52 (105x105) | 2.50      | 2.71     | 2.79          | 3.63 |
| 64 (129x129) | 3.14      | 3.10     | 3.39          | 4.58 |
| growth 13->129 | +1.95   | +2.13    | +2.31         | +3.10 |

## Findings
1. **mgm_norot LOCAL is SIZE-INVARIANT at extreme OOD.** Trained on <=25x25, LOCAL error is FLAT at 0.11 from
   105x105 to 129x129 (~10x training size), ~3x better than GRU (0.31). GLOBAL also stays ~30% better than
   GRU at every OOD size. The motion-gated-memory arch holds up far out of distribution.
2. **idea-B (periodic memory gate) is an honest NEGATIVE.** mgm_grid ~ mgm_norot everywhere (GLOBAL 3.10 vs
   3.14 at n64, within noise); doubling memory (M128) is slightly worse. The plain Gaussian motion-gate
   already captures the benefit; the multi-scale periodic code is redundant here. (Partly vindicates the
   user's own earlier doubt that "toroidal might be wrong" for this task.)
3. **GLOBAL still grows for every arch** — the R3 sqrt single-gauge-anchor floor, fundamental, not a failure.
   mgm_norot grows slowest. For the reframed goal (reach the exit without getting lost) GLOBAL metric
   precision is not required; size-invariant LOCAL + the learned loop-closure memory is.

## Verdict on the learned-arch deliverable
`mgm_norot` (MLP encoder + GRU local integration + bounded motion-gated memory, 100k params) is a working
new architecture for OOD-size state tracking: size-invariant navigable accuracy out to ~10x training size,
beating GRU/Transformer baselines that memorize and transfer worse. The motion-gated memory (the R6-C
diagnosis, made a soft differentiable inductive bias) is the validated key component. The global-metric
residual is the fundamental gauge floor, not an arch deficiency.

## Next options
A. Connect MGM to the NAVIGATION task (user's reframed goal): MGM as the state estimator in an explore-to-
   exit agent; measure reach-rate + redundancy ("not getting lost") vs size. Closes the loop to the goal.
B. Training trick to push global: size-consistency regularization (penalize behavior drift across train
   sizes), or an auxiliary loop-closure loss on the memory.
C. Accept mgm_norot as the Stage-1 deliverable and write it up (the size-invariant local tracker + ablations).
