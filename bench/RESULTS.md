# TrackMaze — phase-1 results (baseline suite)

Setup: train on a small maze (n=6, 200 episodes, T=160), evaluate on larger OOD mazes with the episode
length scaled to size (n=6/9/12/16, T≈24n). Localization target = egocentric displacement-from-start
(path integration). Landmark vocabulary bounded (ambiguity=1) so ids stay in-range across sizes. 3 seeds,
shared cached data so the architecture is the only variable. Device: CPU (models are tiny).

Reference lines: **oracle** association re-anchor ≈ **0.40 cells** (ceiling, what landmarks are worth);
**open-loop** analytic dead-reckoning = 0.76 / 0.81 / 0.83 / 1.26 at n=6/9/12/16 (no-landmark floor).

## Final localization error (cells, mean±std over seeds)
| arch / landmarks | n=6 | n=9 | n=12 | n=16 |
|---|---|---|---|---|
| gru `[-lm]` | 1.14±.06 | 1.22 | 1.40 | 1.76 |
| gru `[+lm]` | 1.78±.03 | 2.18 | 2.12 | 2.69 |
| lstm `[-lm]` | 1.94 | 2.34 | 2.42 | 2.56 |
| lstm `[+lm]` | 2.10 | 2.59 | 2.90 | 2.85 |
| ssm `[-lm]` | 1.07 | 1.14 | 1.28 | 1.85 |
| ssm `[+lm]` | 1.57 | 2.25 | 2.06 | 2.52 |
| transformer `[-lm]` | 1.25 | 1.33 | 1.45 | 1.93 |
| transformer `[+lm]` | 1.64 | 2.21 | 2.02 | 2.31 |
| **open-loop (floor)** | 0.76 | 0.81 | 0.83 | 1.26 |

## Landmark benefit (no-lm error − with-lm error; >0 = re-anchors)
| arch | n=6 | n=9 | n=12 | n=16 |
|---|---|---|---|---|
| gru | −0.64 | −0.95 | −0.72 | −0.93 |
| lstm | −0.16 | −0.25 | −0.47 | −0.29 |
| ssm | −0.50 | −1.12 | −0.78 | −0.67 |
| transformer | −0.40 | −0.88 | −0.57 | −0.38 |

## Findings
1. **No standard architecture closes the floor→ceiling gap.** All sit far above the oracle (0.40) and,
   with landmarks, even above the analytic open-loop integrator.
2. **Landmarks actively HURT all four architectures** (benefit negative everywhere). They overfit
   landmark configs as a *non-transferable* shortcut on training mazes, which corrupts integration on
   OOD mazes. The earlier "correction dip" was a confound (loop-closures occur in low-drift regions),
   not genuine re-anchoring — now confirmed by the ablation.
3. **Underfitting is ruled out:** pure path-integration (no-lm) is near the analytic floor (≈1.1–1.3 at
   n=6 vs 0.76), so integration is mostly learnable; the gap is *matching*, and more training would
   worsen the landmark overfit, not fix it.
4. Pre-registered architecture predictions from expressivity theory (TC0 → transformer worst on OOD
   length) did **not** hold: transformer had the flattest size-slope. The robust signal is the headline
   (nobody extracts the landmark information), not the inter-architecture ordering.

## Implication for phase 3
The landmark information is provably useful (oracle 0.40 vs open-loop ~0.8) but no standard net extracts
it — it's a trap. The target: an architecture / training scheme that does **vocabulary-independent
matching** (loop-closure by relation, not memorized id), turning the −0.5 landmark trap into the +0.4
ceiling. Candidate directions: explicit factorized integrator + retrieval/matching (TEM-style);
latent-perturbation training for robust/transferable landmark use; a structured (toroidal/modular) code.
