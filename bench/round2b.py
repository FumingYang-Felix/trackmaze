"""Round 2-B: LEARNED constant-rate local re-anchor (no oracle/true position). A trained integrator drifts;
at eval we run an online LOCAL landmark map: each landmark seen (relative offset extracted from the omni
view) is matched by PROXIMITY to a recently-stored landmark (local data association, aliasing-robust); a
match re-anchors a running offset. We report GLOBAL error (displacement-from-start) AND LOCAL error
(relative displacement over a 30-step window) vs growing maze size.

Derived expectation: LOCAL error stays FLAT with size (constant-rate re-anchor bounds local drift); GLOBAL
error still grows (a fixed-memory local map can't keep an unbounded space globally consistent). Flat LOCAL
error = the navigable, size-invariant capability; global metric needs loop-closure/anchor (separate).
"""
import argparse, math, numpy as np, torch
from generate_allo import gen_allo
from contract_tracker import CT, tens, train

MAXD, KO = 24.0, 32
_ang = np.array([2*math.pi*i/KO for i in range(KO)]); _dirs = np.stack([np.cos(_ang), np.sin(_ang)], 1)

def reanchor(mu, geo, lm, gain=0.3, thr=1.2, forget=60):
    T = mu.shape[0]; lms = []; offset = np.zeros(2, np.float32); out = np.zeros((T, 2), np.float32)
    for t in range(T):
        est = mu[t] + offset
        for i in np.where(lm[t] > 0.5)[0]:
            rel = geo[t, i] * MAXD * _dirs[i]; obs_abs = est + rel
            if lms:
                d = np.array([np.linalg.norm(obs_abs - p) for p, _ in lms]); j = int(d.argmin())
                if d[j] < thr:
                    offset = offset + gain * (lms[j][0] - obs_abs); est = mu[t] + offset
                    lms[j] = (0.9*lms[j][0] + 0.1*(est+rel), t); continue
            lms.append((est + rel, t))
        lms = [(p, lt) for p, lt in lms if t - lt < forget]
        out[t] = mu[t] + offset
    return out

def gerr(pred, true): return np.linalg.norm(pred - true, axis=2)[:, -1].mean()
def lerr(pred, true, W=30):
    rp = pred[:, W:] - pred[:, :-W]; rt = true[:, W:] - true[:, :-W]
    return np.linalg.norm(rp - rt, axis=2).mean()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gain", type=float, default=0.3); ap.add_argument("--thr", type=float, default=1.2); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    G = dict(ambiguity=1, canon="cmd", rot_noise=0.09, loop=0.3)
    tr = gen_allo(n=6, n_eps=200, T=160, seed=1, **G); x, y = tens(tr)
    m = CT(x.shape[2]); train(m, x, y, 0.0, a.epochs)
    print(f"{'n':>4} | {'INT global':>10} {'+RA global':>10} | {'INT local':>9} {'+RA local':>9}")
    rows = []
    for n in (6, 12, 20, 28, 40):
        te = gen_allo(n=n, n_eps=30, T=24*n, seed=7, **G); xt, _ = tens(te)
        with torch.no_grad(): mu = m(xt).numpy()
        mura = np.stack([reanchor(mu[e], te["canon_geo"][e], te["canon_lm"][e], a.gain, a.thr) for e in range(mu.shape[0])])
        D = te["disp"]; gi, gr, li, lr = gerr(mu, D), gerr(mura, D), lerr(mu, D), lerr(mura, D)
        rows.append((n, li, lr)); print(f"{n:>4} | {gi:10.2f} {gr:10.2f} | {li:9.2f} {lr:9.2f}")
    grow_int = rows[-1][1] - rows[0][1]; grow_ra = rows[-1][2] - rows[0][2]
    print(f"RESULT seed={a.seed} gain={a.gain} thr={a.thr} LOCAL_grow(n6->n40) INT={grow_int:+.2f} +RA={grow_ra:+.2f} "
          f"local_n40 INT={rows[-1][1]:.2f} +RA={rows[-1][2]:.2f}")
