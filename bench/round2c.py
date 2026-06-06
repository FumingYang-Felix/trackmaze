"""Round 2-C: confirm the integrator's LOCAL error is FLAT across GROWING maze sizes at convergence (no
re-anchor -- 2-B showed it hurts). LOCAL = error in relative displacement over a window (navigable
accuracy). GLOBAL = displacement-from-start error (grows; memory-limited). If LOCAL is flat from 13x13 to
121x121, size-invariant local tracking is achieved by continuous-observation integration alone.
"""
import argparse, numpy as np, torch
from generate_allo import gen_allo
from contract_tracker import CT, tens, train

def gerr(p, t): return np.linalg.norm(p - t, axis=2)[:, -1].mean()
def lerr(p, t, W=30):
    rp = p[:, W:] - p[:, :-W]; rt = t[:, W:] - t[:, :-W]; return np.linalg.norm(rp - rt, axis=2).mean()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    G = dict(ambiguity=1, canon="cmd", rot_noise=0.09, loop=0.3)
    tr = gen_allo(n=6, n_eps=200, T=160, seed=1, **G); x, y = tens(tr)
    m = CT(x.shape[2]); train(m, x, y, 0.0, a.epochs)
    glob, loc = [], []
    for n in (6, 12, 20, 28, 40, 52):
        te = gen_allo(n=n, n_eps=30, T=24*n, seed=7, **G); xt, _ = tens(te)
        with torch.no_grad(): mu = m(xt).numpy()
        g, l = gerr(mu, te["disp"]), lerr(mu, te["disp"]); glob.append((n, g)); loc.append((n, l))
    print(f"RESULT seed={a.seed} LOCAL={[(n, round(l,2)) for n,l in loc]} GLOBAL={[(n, round(g,2)) for n,g in glob]} "
          f"local_grow(n6->n52)={loc[-1][1]-loc[0][1]:+.2f} global_grow={glob[-1][1]-glob[0][1]:+.2f}")
