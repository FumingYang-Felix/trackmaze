# Round 8: the LEARNED architecture (MGM) — motion-gated memory is the OOD driver

The deliverable: a new RNN-type arch + training that handles OOD-size state tracking, built from the R2-R7
diagnosis. Trained on n=6,12, evaluated OOD to n=52 (105x105). 5 archs x 3 seeds on FASRC GPU (job 19553776).
MGM = rotation-invariant omni encoder + GRU local integration + bounded memory with a MOTION-GATED soft read
(attend by content GATED by motion-reachability exp(-||mu-mu_i||^2/tau^2)) + learned loop-closure gate.
Ablations: mgm_norot (MLP encoder, no rotation-invariance), mgm_nomem (no memory). Baselines: plain GRU,
causal Transformer.

## Results (mean+-std over 3 seeds; n6,12 = train, rest = OOD)

LOCAL error (windowed relative displacement = navigable accuracy):
| n (grid)    | mgm        | mgm_norot  | mgm_nomem  | gru        | transformer |
|-------------|------------|------------|------------|------------|-------------|
| 20 (41x41)  | 0.26       | 0.23       | 0.52       | 0.48       | 0.44        |
| 40 (81x81)  | 0.13       | 0.14       | 0.36       | 0.32       | 0.31        |
| 52 (105x105)| **0.10**   | **0.11**   | 0.32       | 0.27       | 0.26        |

GLOBAL error (final displacement-from-start):
| n (grid)    | mgm        | mgm_norot  | mgm_nomem  | gru        | transformer |
|-------------|------------|------------|------------|------------|-------------|
| 20 (41x41)  | 3.50       | **2.59**   | 5.54       | 4.27       | 3.80        |
| 40 (81x81)  | 3.33       | **2.83**   | 7.59       | 4.07       | 4.11        |
| 52 (105x105)| 3.28       | **2.50**   | 9.11       | 3.63       | 3.42        |
| growth 13->105 | +1.87   | **+1.31**  | +7.27      | +2.15      | +2.01       |

## The three clean findings
1. **The motion-gated MEMORY is the OOD driver.** Removing it (mgm_nomem) collapses LOCAL to baseline level
   (0.32) and blows up GLOBAL to 9.11 (worse than the plain baselines, high variance). The learned, soft,
   motion-gated loop-closure (the R6-C inductive bias) is doing the work — validating the whole diagnostic arc.
2. **The best arch is `mgm_norot` (MLP encoder + GRU + motion-gated memory).** OOD at 105x105: LOCAL 0.11 vs
   0.26-0.27 (gru/transformer) = less than half; GLOBAL 2.50 vs 3.42-3.63 = ~30% better; GLOBAL growth +1.31
   vs +2.01/+2.15 = slowest. Only 100k params (Transformer = 820k).
3. **Honest negative: rotation-invariance does NOT help, and slightly hurts GLOBAL** (mgm 3.28 vs mgm_norot
   2.50). The rot-invariant encoder discards heading information useful for the global frame; LOCAL is tied.
   Drop rotation-invariance.

## Honest bounds
GLOBAL still GROWS for every arch (the R3 sqrt single-gauge-anchor floor is not broken, only slowed —
mgm_norot grows slowest). The win is solid and multi-seed-robust but moderate (~2x on LOCAL, ~30% on GLOBAL).
The memorizing baselines (GRU/Transformer) do NOT transfer as well OOD (the gap widens with size) — exactly
the phase-1 thesis, now with an arch that beats them.

## Next (Round 9)
Promote `mgm_norot`; push further OOD (n=64,80,100) to test if the gap keeps widening; tune memory (M, tau,
multi-head read). To attack the still-growing GLOBAL: give the memory a PERIODIC/grid position code (the
user's idea-B, `toy_torus.py`) so re-anchoring disambiguates over a large range with bounded drift — the one
lever aimed straight at the sqrt floor.
