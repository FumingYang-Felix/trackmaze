"""Positive controls: verify the training pipeline/implementation is correct, so the negative findings
(landmarks hurt OOD; RLC plateaus) are real science, not bugs.

A. ORACLE CUE: append the true displacement to the input. The model must reach ~0 error (trivial copy).
   High error => the head / target / training loop is broken.
B. SAME-MAZE (in-distribution): train and evaluate on the SAME mazes. Landmarks must HELP here
   (memorization works in-distribution). If +lm doesn't beat -lm even in-distribution, landmarks are
   not being used at all => a wiring bug.
C. CONVERGENCE: double epochs + data for the no-lm integrator; if the floor barely moves, it's converged
   (the ~1.1 floor is real, not undertraining).
"""
import numpy as np, torch
from diversity_sweep import gen_pool, final_err
from train_eval import encode, train, cached, AMB
from archs import build, GRUTracker
from metrics import per_step_error

torch.manual_seed(0); np.random.seed(0)
tr = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)

# ---- A: oracle cue ----
xb, y = encode(tr, False); xo = torch.cat([xb, y], dim=2)            # append the answer
mA = GRUTracker(xo.shape[2]); train(mA, xo, y, 40, "cpu")
te = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)
xt, _ = encode(te, False); xt = torch.cat([xt, torch.from_numpy(np.asarray(te["disp"], np.float32))], dim=2)
with torch.no_grad(): eA = per_step_error(mA(xt).numpy(), te["disp"])[:, -1].mean()
print(f"A. oracle-cue error = {eA:.3f}   (PASS if ~0; FAIL/bug if large)")

# ---- B: same-maze, landmarks should help ----
pool = gen_pool(n=6, n_distinct=8, reps=8, T=160, ambiguity=AMB, base_seed=1)
res = {}
for use_lm in (True, False):
    torch.manual_seed(0); x, yy = encode(pool, use_lm); mB = build("gru", x.shape[2]); train(mB, x, yy, 60, "cpu")
    with torch.no_grad(): res[use_lm] = per_step_error(mB(x).numpy(), pool["disp"])[:, -1].mean()
print(f"B. in-distribution (8 mazes):  +lm = {res[True]:.2f}   -lm = {res[False]:.2f}   "
      f"(PASS if +lm clearly LOWER; landmarks usable in-dist)")

# ---- C: convergence of the no-lm floor ----
floors = {}
for ne, ep in [(200, 60), (600, 120)]:
    big = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1) if ne == 200 else \
          gen_pool(n=6, n_distinct=600, reps=1, T=160, ambiguity=AMB, base_seed=5)
    torch.manual_seed(0); x, yy = encode(big, False); mC = build("gru", x.shape[2]); train(mC, x, yy, ep, "cpu")
    floors[(ne, ep)] = final_err(mC, te, False)
print(f"C. no-lm floor @ n=12:  200ep/60 = {floors[(200,60)]:.2f}   600ep/120 = {floors[(600,120)]:.2f}   "
      f"(PASS if ~stable = converged, not undertrained)")
