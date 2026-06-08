# Round 13 — the REAL Stage-2: a LEARNED navigation policy that actually walks the maze, OOD-size

The user's ask: "我们甚至还没有进行 stage2 的 arch 和训练… 就是真的走迷宫" — a fully *learned* policy that drives the
maze closed-loop (no oracle low-level execution), trained small → tested OOD-large. Earlier nav rounds (R11/R14)
solved completeness with a *hand-coded* DFS + oracle execution; R13 is the learned version.

## The claim under test (from R12)
A learned navigation policy generalizes the navigation strategy in SIZE **iff** it has the right inductive bias:
1. a **drift-free observation frame** — canonicalize obs by the grid-CORRECTED heading (fine part pinned to the
   allothetic freq-4 wall cue), not the drifting command odometry; and
2. a **spatial memory indexed by drift-free topological coords** that grows with explored area.
End-to-end policies that integrate a single drifting frame collapse OOD.

## Why behaviour cloning, not RL
Sparse-reward PPO is the wrong tool here: `rl_nav.py` was run once (job 19686703) and FAILED — reach < 0.1 after
250/2500 iters, then walltime-killed; never produced results. A corrected recurrent-PPO (`nav_ppo.py`, fixing a
flat-concatenation-GRU ratio bug) still hits the sparse-reward / entropy-collapse wall (reach < 0.12 local).
→ **BC** (`nav_bc.py`): clone a *realizable, frame-consistent* frontier-DFS teacher (reaches the exit with
bounded redundancy, size-invariant, start-frame priority so labels are a function of the observable state) into
each arch; **eval closed-loop OOD** (the policy drives from egocentric obs alone — still "真的走迷宫"). BC holds the
training signal fixed and varies only the arch → clean isolation of the inductive bias. DART noise-injection in
the teacher (off-path recovery demos) fixes closed-loop compounding error.

## Archs (same teacher / mazes; the contribution axis)
| arch | obs frame | memory | role |
|------|-----------|--------|------|
| `gru_cmd`  | command heading (DRIFTS) | GRU hidden (fixed) | e2e baseline |
| `gru_corr` | grid-corrected (drift-free) + grid cue | GRU hidden (fixed) | frame lever |
| `tf_corr`  | grid-corrected + grid cue | causal attention (∝ history) | capacity baseline |
| `ours`     | grid-corrected + grid cue | + 3×3 spatial-memory window on drift-free topo (grows w/ area) | +memory lever |

## Results

### First local round (400 trajs, 30 ep, NO DART) — directional confirmation
reach-rate, train sizes 5,7 → OOD 12,20:
- `ours`:    n5 0.62 · n7 0.38 · n12 0.25 · n20 0.00
- `gru_cmd`: n5 0.00 · n7 0.00 · n12 0.00 · n20 0.00  (collapses even in-distribution)
→ structure helps massively; but closed-loop BC compounding error caps `ours` at 62% in-dist → added DART.

### DART + larger run (local validation, then cluster 4×3 seeds)
DART (eps=0.08) noise-injection fixes closed-loop compounding error. Cluster job 19970191 (1500 trajs, 60 ep,
24 eval mazes, up to n=40, 4 archs × 3 seeds) — **the FRAME lever is CLEAN and STRONG** (reach-rate, mean over
seeds, n=5/7/12/20/28/40):
- `gru_cmd`  (drifting frame): ~0.05 / 0.07 / 0.04 / **0.00 / 0.00 / 0.00** — CATASTROPHIC OOD collapse, all seeds.
- `gru_corr` (grid-corrected drift-free frame): ~0.56 / 0.58 / 0.38 / ~0.33 / … — generalizes OOD, path ratio 1.5–3.6.
→ **Correcting the egocentric frame with the allothetic grid-mod-90 cue is what enables OOD-size navigation; the
end-to-end drifting-frame policy fails catastrophically.** This is the "structure beats e2e" thesis (STORY §5,
shown there for heading ESTIMATION) extended to closed-loop BEHAVIOR/navigation — and it is grounded in the
bounded-error theory (drifting frame = unbounded OOD; allothetic anchor = bounded/size-invariant).
**FINAL exploration numbers (mean over 3 seeds, reach-rate; train = 11,15px) — `nav_ood_figure.png`:**
| size px | gru_cmd (drift frame) | gru_corr (drift-free frame) | tf_corr (TF, drift-free) | OURS (drift-free + memory) |
|---------|-----------------------|-----------------------------|--------------------------|----------------------------|
| 11 (tr) | 0.06 | 0.56 | 0.18 | **0.75** |
| 15 (tr) | 0.07 | 0.58 | 0.08 | **0.60** |
| 25      | 0.04 | 0.38 | 0.01 | **0.48** |
| 41      | 0.00 | 0.28 | —    | 0.29 |
| 57      | 0.00 | 0.21 | —    | 0.12 |
| 81 (~6×)| 0.00 | 0.00 | —    | **0.12** |
Path redundancy (steps/teacher): OURS ~1.0–2.0 (most efficient) < gru_corr ~1.5–4.5 < gru_cmd ~3–8.

### STRONG run = the HEADLINE (3000 trajs, 100 ep, 3 seeds; `nav_ood_strong.png`)
Higher-fidelity training LIFTS absolute reach and SHARPENS the dissociation (reach-rate, mean; 11/15/25/41/57/81px):
- `gru_cmd` (drifting frame): 0.17 / 0.07 / 0.04 / **0.01 / 0.01 / 0.01** — collapses.
- `gru_corr` (drift-free grid-corrected frame): **0.78** / 0.64 / 0.46 / 0.49 / 0.32 / 0.21 — ~0.8 in-dist, graceful to ~6×.
- `ours` (+memory): 0.78 / 0.71 / 0.53 / 0.49 / 0.29 / 0.17 — ≈ gru_corr (memory washes out for exploration).
- `gru_compass` (+ allothetic GLOBAL heading cue, resolves the Z₄ quadrant — STORY §7): 0.68 / 0.53 / 0.35 / 0.43
  / 0.21 / **0.40** — slightly lower in-dist but the **only arch that SUSTAINS reach at the extreme 81px (~6×)**
  (0.40 vs 0.17–0.21), where the grid-corrected frame's residual quadrant drift finally bites.
**→ the FRAME is THE lever** (drift-free ≈0.78 vs drifting ≈0.15 in-dist; ≈0.2–0.4 vs ~0 at 6× OOD); the MEMORY
lever WASHES OUT for exploration at high training (ours ≈ gru_corr). The COMPASS (allothetic global cue) does NOT
give flat size-invariance (exploration task difficulty itself scales with size) but it SUSTAINS extreme-OOD nav
where grid-only decays — the §7 "allothetic orientation resolves the quadrant at scale" principle, in behaviour
(honest: suggestive + noisy, not a dramatic flat line). Clean headline: *an end-to-end policy that integrates a
drifting egocentric frame fails catastrophically OOD-size; the allothetic grid-anchoring correction gives
graceful OOD navigation, and a global allothetic cue sustains it at the largest scales.*

### Maze HOMING (`nav_home.py --pos`, 3 seeds) — corroborates; memory matters MORE here, but low/noisy
return-to-start rate (mean; 11/15/25/41/57px): gru_cmd 0.17/0.17/0.15/0.05/0.10 · gru_corr 0.32/0.30/0.15/0.08/0.08
· tf_corr 0.25/0.05/0.07/0.05/0.03 · **ours 0.50/0.23/0.18/0.17/0.02**. Ordering ours > gru_corr > tf_corr ≈
gru_cmd in-dist → homing (which REQUIRES the cognitive map to find the way back) shows the **memory lever** that
exploration washes out. But absolute reach is LOW and OOD is noisy → supporting evidence, not a headline. (DAgger
made it worse; not pursued.) Contrast with open-arena metric homing (itwm pi_home): that COMPOUNDS OOD; topological
maze homing is bounded-but-low here.

**Mechanism (honest, `frame_drift.py`).** Per-step heading-frame error during a scripted DFS walk, mod-90, each
frame vs its own anchor, flat across size: cmd (idiothetic, vs start) ≈ 22° = already RANDOM/lost (the rot-noise
drift saturates past 45° within the first maze, at ALL sizes); corr (allothetic grid, vs world) ≈ 16° = pinned,
flat. So the cmd frame loses the fine heading *fast* (why gru_cmd is bad even in-distribution), while the
grid-corrected frame stays locked at every size (why gru_corr/ours work). The OOD *decline* of the good archs is
NOT fine-heading drift (flat) — it's the discrete QUADRANT slip over long trajectories (STORY §4 Z₄) + harder
exploration at scale. (frame_drift was a side-analysis; the behavioral dissociation + STORY §3–§4 theory is the
mechanism — not featured as a main figure.)

**Read.** (1) FRAME lever = large & clean: gru_cmd collapses to 0 from 41px; the drift-free archs generalize.
(2) MEMORY lever = real but modest: OURS is best at train+near-OOD (0.75/0.60/0.48) and the MOST path-efficient,
and is uniquely non-zero at 81px (~6× train) where gru_corr finally fails — spatial memory extends the size
limit; mid-range (57px) is within noise of gru_corr. (3) CAPACITY is not the lever: tf_corr (same drift-free obs)
collapses by 25px. Absolute reach is moderate (one-shot BC of a ~500-step explorer is hard) — the scientific
point is the DISSOCIATION + graceful-vs-catastrophic OOD degradation, grounded in the bounded-error theory.

**The rigorous controlled comparison is `gru_cmd` vs `gru_corr`** — IDENTICAL GRU arch, IDENTICAL training, the
ONLY difference is the obs-frame canonicalization (drifting command heading vs drift-free grid-corrected heading).
gru_cmd collapses, gru_corr generalizes → the frame correction is causally the lever. `tf_corr` (same drift-free
obs, Transformer instead of GRU) also collapses (0.17/0.08/0.00…) → attention/capacity is NOT a substitute for
the right recurrent inductive bias (and may be under-tuned for BC; reported as a baseline, not the controlled arm).

**DAgger (homing) — NEGATIVE / not pursued.** On-policy relabeling (`--dagger_rounds`) made homing WORSE
(ours 0.19/0.12/0.06/0.12) because the sampled untrained policy wanders OFF the discovered graph → BFS returns
None → fallback labels flood the aggregated set with junk. Would need: greedy rollout, early-abort when lost,
drop off-graph steps, bound the DAgger data fraction. Deprioritized — the plain-BC frame-lever result is clean.

## Maze HOMING (nav_home.py) — the cleaner cognitive-map test
Forced outbound DFS (observe only) → return-to-start closed loop; teacher = unique BFS shortest path back
(unambiguous BC labels). Teacher home-success = 1.00.

### Control: NO explicit position signal (memory window only)
home-rate n5/7/12/20: ours 0.58/0.50/0.33/0.42 · gru_corr 0.58/0.42/0.42/0.17 · gru_cmd 0.42/0.42/0.50/**0.08**.
→ At the largest OOD (n20) the dissociation emerges (ours 0.42 ≫ gru_cmd 0.08), but it is WEAK/NOISY because no
arch has a clean homing cue — the visited-cell memory window alone is ~noise for "which way is home", so the
simplest obs (gru_cmd) even wins at n12. **Diagnosis: homing needs an explicit START-RELATIVE position estimate.**

### Fix: `--pos` — feed each arch its OWN home-bearing (drift-free for corr/ours, drifting for cmd)
`NavState` now integrates a start-relative metric position in the arch's own canonicalization frame and exposes
`home_signal()` = (cos,sin of bearing-to-start rel current heading, saturated distance). This isolates
INTEGRATION QUALITY: gru_cmd's bearing drifts → wrong direction OOD → collapses; corr/ours bearings are
drift-free → home OOD; ours adds the spatial map for wall-routing → should be best & flat. (local validation +
cluster `submit_navhome.sh` IN PROGRESS.)

## Honest expected shape / limits
`ours` should stay high & ~flat across size then slowly decline (the topo quadrant can drift over very long
trajectories — the same Z₄ limit as the estimation story); `gru_cmd` collapses immediately; `gru_corr`/`tf_corr`
fall in between. The point is the **dissociation** (drift-free frame + growing memory are each necessary),
complementing the estimation phase-diagram with a constructive, *positive*, size-invariant navigation result.
