"""Test 2: the decisive fork. Sweep #distinct training worlds x model capacity, and watch the LANDMARK
BENEFIT (no-lm error minus with-lm error). If benefit crosses POSITIVE at high diversity / low capacity,
the relational re-anchoring algorithm is LEARNABLE FROM DATA (no special arch needed). If it saturates at
~0 from below, it is NOT learnable by a vanilla GRU and needs an inductive bias (CADER / ToroTrack).

(Honest caveat: at the 1024-world point total data grows since we can't repeat; that only pushes harder
toward generalization, so a crossing is still meaningful.)
"""
import numpy as np, torch
from diversity_sweep import gen_pool, final_err
from train_eval import encode, train, cached, AMB
from archs import GRUTracker

WORLDS = [16, 64, 256, 1024]; CAPS = [32, 128]; T = 160
te = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)

print("generating world pools (once each) ...")
pools = {}
for nd in WORLDS:
    total = max(256, nd); pools[nd] = gen_pool(n=6, n_distinct=nd, reps=max(1, total//nd), T=T, ambiguity=AMB, base_seed=1)
    print(f"  pool {nd} worlds ready")

print("\nLANDMARK BENEFIT (no-lm minus with-lm) at n=12 — >0 means landmarks HELP (shortcut beaten)")
print(f"{'capacity':>9} | " + " ".join(f"{('w='+str(w)):>10}" for w in WORLDS))
for cap in CAPS:
    row = f"{('h='+str(cap)):>9} | "
    for nd in WORLDS:
        r = {}
        for use_lm in (True, False):
            torch.manual_seed(0); np.random.seed(0)
            x, y = encode(pools[nd], use_lm); m = GRUTracker(x.shape[2], h=cap); train(m, x, y, 50, "cpu")
            r[use_lm] = final_err(m, te, use_lm)
        row += f"{r[False]-r[True]:+10.2f}"
    print(row)
print("\nrows = capacity, cols = #worlds. Crossing into POSITIVE = relational use is learnable from data.")
