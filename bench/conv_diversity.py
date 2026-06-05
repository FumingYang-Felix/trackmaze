"""Convergent re-run of the diversity fork (one config per process; launch in parallel).
Fixes the undertraining caught by Control C: 120 epochs (vs 50), cached world pools, multiple seeds.
Prints one greppable RESULT line: landmark benefit (no-lm minus with-lm) at n=12 for (#worlds, seed).
"""
import argparse, os, numpy as np, torch
from diversity_sweep import gen_pool, final_err
from train_eval import encode, train, cached, AMB, CACHE
from archs import GRUTracker

def get_pool(nworlds):
    p = os.path.join(CACHE, f"pool_w{nworlds}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
    total = max(384, nworlds)
    ds = gen_pool(n=6, n_distinct=nworlds, reps=max(1, total // nworlds), T=160, ambiguity=AMB, base_seed=1)
    np.savez(p, **{k: v for k, v in ds.items()}); return ds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--cap", type=int, default=128)
    a = ap.parse_args()
    pool = get_pool(a.worlds)
    te = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)
    r = {}
    for use_lm in (True, False):
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        x, y = encode(pool, use_lm); m = GRUTracker(x.shape[2], h=a.cap); train(m, x, y, a.epochs, "cpu")
        r[use_lm] = final_err(m, te, use_lm)
    print(f"RESULT worlds={a.worlds} seed={a.seed} benefit={r[False]-r[True]:+.3f} (+lm {r[True]:.2f} / -lm {r[False]:.2f})")
