"""Round 6-B: ROBUST loop closure (Dynamic Covariance Scaling) vs R5's naive gating. The descriptor proposes
many candidate closures (~80% false under local aliasing, R4-C). A robust back-end jointly optimizes the
pose graph while DOWN-WEIGHTING closures inconsistent with the trajectory: a false closure between two
look-alike places implies a loop whose odometry doesn't sum to ~0, so its residual is large and DCS switches
it off. This is the standard SLAM answer to aliasing that R5's per-pair gate lacked.

Regime: oracle heading + elevated translation odometry noise (so dead-reckon drifts enough that loop closure
matters and robust-vs-naive is distinguishable; heading/SE(2) is the separate R6-A piece). Translation-only
pose graph (x,y decouple). DCS weight w_k = clamp((2*phi/(phi+s_k))^2, 0,1), s_k = closure residual^2.

Compares global mean error vs size for:
  dead    : odometry only (no closure)
  naive   : all gated descriptor candidates as hard closures (R5 style)
  robust  : same candidates, DCS-downweighted
  oracle  : oracle place-identity closures (upper bound)
and reports closure precision and the effective weight DCS keeps on true vs false closures.
"""
import argparse, math, sys, numpy as np, torch
import scipy.sparse as sp, scipy.sparse.linalg as spla
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni
from round3a import cell_graph, dfs_walk, wrap, motion
from round4b import RotInv, feats, build_pairs, train_encoder, gen_dfs

sys.setrecursionlimit(2_000_000)


def rollout(env, n, anoise):
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    true0 = np.array([env.px, env.py], float); est = np.zeros(2)
    T, E, C, L = [], [], [], []; last = None

    def rec():
        g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
        T.append(np.array([env.px, env.py], float) - true0); E.append(est.copy())
        C.append(int(env.py) * 100000 + int(env.px)); L.append(np.concatenate([g, l]))

    rec(); last = (int(env.px), int(env.py))
    for (cx, cy) in walk[1:]:
        wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(40):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            err = wrap(math.atan2(dy, dx) - env.ang)
            if abs(err) > 0.20: a = 3 if err > 0 else 2
            else:
                a = 0; step = np.array([math.cos(env.ang), math.sin(env.ang)]) * 0.22
                est = est + step + np.random.normal(0, anoise, 2)         # elevated translation odo noise
            motion(env, a)
            cell = (int(env.px), int(env.py))
            if cell != last: rec(); last = cell
    return np.array(T), np.array(E), np.array(C), np.stack(L).astype(np.float32)


def candidates(est, desc, R, thr):
    N = len(est); inv = 1.0 / R; buckets = defaultdict(list); key = np.floor(est * inv).astype(int); out = []
    for i in range(N): buckets[(key[i, 0], key[i, 1])].append(i)
    R2 = R * R
    for i in range(N):
        bx, by = key[i]; cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): cand += buckets.get((bx + dx, by + dy), [])
        for j in cand:
            if j <= i + 2: continue
            d = est[i] - est[j]
            if d @ d < R2 and float(desc[i] @ desc[j]) > thr: out.append((i, j))
    return out


def solve(est, loops, weights=None, w_loop=4.0):
    N = len(est); L = sp.lil_matrix((N, N)); bx = np.zeros(N); by = np.zeros(N)
    for i in range(N - 1):
        d = est[i + 1] - est[i]
        L[i, i] += 1; L[i + 1, i + 1] += 1; L[i, i + 1] -= 1; L[i + 1, i] -= 1
        bx[i] -= d[0]; bx[i + 1] += d[0]; by[i] -= d[1]; by[i + 1] += d[1]
    for k, (i, j) in enumerate(loops):
        w = w_loop * (1.0 if weights is None else weights[k])
        L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w     # closure: x_i == x_j
    free = list(range(1, N)); Lf = L.tocsr()[free][:, free]
    xs = np.zeros(N); ys = np.zeros(N)
    xs[free] = spla.spsolve(Lf, bx[free]); ys[free] = spla.spsolve(Lf, by[free])
    return np.stack([xs, ys], 1)


def dcs_solve(est, loops, phi=2.0, iters=8, w_loop=4.0):
    if not loops: return solve(est, []), np.array([])
    w = np.ones(len(loops))
    for _ in range(iters):
        X = solve(est, loops, w, w_loop)
        s = np.array([np.sum((X[i] - X[j]) ** 2) for i, j in loops])      # closure residual^2
        w = np.clip((2 * phi / (phi + s)) ** 2, 0.0, 1.0)                 # DCS down-weighting
    return solve(est, loops, w, w_loop), w


def merr(X, true): return float(np.mean(np.linalg.norm(X - true, axis=1)))


def run_size(n, enc, R, thr, anoise, n_mazes=6, seed0=12000):
    out = {"dead": [], "naive": [], "robust": [], "oracle": []}; precs = []; wsep = []
    for m in range(n_mazes):
        np.random.seed(seed0 + m)
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9,
                           action_noise=0.03, rot_noise=0.09)
        env.reset()
        true, est, cell, view = rollout(env, n, anoise)
        with torch.no_grad(): desc = enc(torch.tensor(view)).numpy()
        cand = candidates(est, desc, R, thr)
        first = {}; oloops = []
        for i, c in enumerate(cell):
            if c in first: oloops.append((first[c], i))
            else: first[c] = i
        if cand:
            tp = np.array([cell[i] == cell[j] for i, j in cand]); precs.append(tp.mean())
        out["dead"].append(merr(solve(est, []), true))
        out["naive"].append(merr(solve(est, cand), true))
        Xr, w = dcs_solve(est, cand)
        out["robust"].append(merr(Xr, true))
        out["oracle"].append(merr(solve(est, oloops), true))
        if cand and len(w):
            tp = np.array([cell[i] == cell[j] for i, j in cand])
            if tp.any() and (~tp).any(): wsep.append((w[tp].mean(), w[~tp].mean()))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    wt = np.mean([a for a, b in wsep]) if wsep else 0; wf = np.mean([b for a, b in wsep]) if wsep else 0
    return res, float(np.mean(precs)) if precs else 0.0, wt, wf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=4.0); ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--anoise", type=float, default=0.12); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--mazes", type=int, default=6); ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    a = ap.parse_args()
    print(f"Round 6-B: robust (DCS) loop closure. train enc n=6,12; eval OOD. R={a.R} thr={a.thr} anoise={a.anoise}")
    tr6, tr12 = gen_dfs(6, 4, "true", 7000), gen_dfs(12, 3, "true", 7100)
    tr = {k: np.concatenate([tr6[k], tr12[k]]) for k in ("geo", "lm", "cell")}
    tr["ep"] = np.concatenate([tr6["ep"], 1000 + tr12["ep"]])
    enc = train_encoder(RotInv(), feats(tr), build_pairs(tr), epochs=a.epochs)
    print("  encoder trained\n")
    print(f"{'n':>4} {'grid':>8} | {'cand_prec':>9} {'w_true':>6} {'w_false':>7} | {'dead':>6} {'naive':>7} {'robust':>7} {'oracle':>7}")
    g = {k: [] for k in ("dead", "naive", "robust", "oracle")}
    for n in a.sizes:
        res, p, wt, wf = run_size(n, enc, a.R, a.thr, a.anoise, n_mazes=a.mazes)
        for k in g: g[k].append(res[k])
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {p:9.3f} {wt:6.2f} {wf:7.2f} | {res['dead']:6.2f} {res['naive']:7.2f} {res['robust']:7.2f} {res['oracle']:7.2f}")
    print("\nGROWTH (n_min -> n_max):")
    for k in g: print(f"  {k:>7}: {g[k][0]:.2f} -> {g[k][-1]:.2f}  ({g[k][-1]-g[k][0]:+.2f})")
