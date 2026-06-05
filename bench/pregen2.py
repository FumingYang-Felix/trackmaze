"""Pre-generate the caches train_rlc needs (train_n6 + all eval sizes), once, before the array."""
from train_eval import cached, AMB, TRAIN_N, TRAIN_T, TRAIN_EPS, EVAL_SIZES
cached(f"train_n{TRAIN_N}_T{TRAIN_T}_a{AMB}", n=TRAIN_N, n_eps=TRAIN_EPS, T=TRAIN_T, ambiguity=AMB, seed=1)
for n, T in EVAL_SIZES:
    cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
print("rlc caches ready:", EVAL_SIZES)
