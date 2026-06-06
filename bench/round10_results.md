# Round 10 (D — strengthen the architecture): modest gains, mgm_v2_big is the strengthened deliverable

Strengthen mgm_norot with: learned key/query projections for the memory read, learned write-importance (R3,
as a log bias), iterative loop-closure correction (n_corr), and more capacity. Add LSTM baseline. 4 archs x
3 seeds, eval to n=64 (129x129), FASRC job 19627313.

## Results (mean+-std, 3 seeds; train = n6,12)
| n (grid)     | mgm_norot (100k) | mgm_v2 (201k) | mgm_v2_big (303k) | lstm (101k) |
|--------------|------------------|---------------|-------------------|-------------|
| LOCAL  52    | 0.10             | 0.11          | **0.09**          | 0.47        |
| LOCAL  64    | 0.11             | 0.12          | **0.09**          | 0.50        |
| GLOBAL 52    | 2.35             | 2.87          | 2.49              | 5.15        |
| GLOBAL 64    | 3.13             | 3.31          | **2.76**          | 6.63        |
| GLOBAL growth 13->129 | +1.98   | +2.10         | **+1.49**         | +4.81       |

## Findings (honest)
1. **mgm_v2_big is the new best** (LOCAL 0.09, GLOBAL 2.76 at 129x129, GLOBAL growth +1.49 — the slowest yet,
   down from mgm_norot's +1.98). Closer to the R3 oracle sqrt-floor.
2. **The gain is mostly CAPACITY, not the new mechanisms per se.** mgm_v2 (medium, kv+iterative but h=192)
   is NOT better than mgm_norot (GLOBAL 3.31 vs 3.13); only the bigger mgm_v2_big (h=256, n_corr=3, M=128)
   helps. Diminishing returns -- the sqrt floor still bounds GLOBAL.
3. **LSTM is the worst tracker** (worse than GRU): GLOBAL 6.63 at n64. The recurrent-baseline ranking is
   GRU > LSTM; both far below the MGM family.

## Verdict
D done. `mgm_v2_big` (303k params) is the strengthened OOD-size state tracker: size-invariant LOCAL (~0.09
out to 10x training) and the slowest GLOBAL growth of any arch. Further arch tweaking has low ROI (sqrt
floor). Proceed to A (navigation): use the trained MGM as an online state estimator in an explore-to-exit
agent, measure reach-rate + redundancy ("not getting lost") vs size.
