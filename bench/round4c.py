"""Round 4-C: the descriptor precision wall. R5 showed a single-omni-view descriptor (AUC~0.91) gives only
~18% loop-closure precision, because a metric gate of radius R contains many candidate cells and AUC 0.91 is
nowhere near enough for k-way discrimination (0.91^k decays fast). Loop closure needs AUC ~0.99+.

Hypothesis: a LOCAL SUBMAP descriptor -- the actual wall+landmark layout in a fixed radius W around the cell
(route-invariant, built from R2's size-invariant LOCAL accuracy) -- is far more distinctive than a single
omni view, and its distinctiveness is size-invariant. Here we measure the CEILING with an ORACLE local
submap (read straight from the true maze) vs size and radius W. If AUC -> ~0.99 at modest W, the descriptor
problem is solvable (a learned version from accurate local tracking can approach it); if even the oracle
submap aliases, the maze is locally too self-similar and no descriptor fixes it.

Also reports k-way TOP-1 retrieval precision: for each revisit, is the true earlier visit the nearest among
all nodes within a metric gate R? -- the quantity loop closure actually needs.
"""
import argparse, math, numpy as np
from collections import defaultdict
from env import TrackMazeEnv
from round3a import cell_graph, dfs_walk, wrap, motion


def rollout_submap(env, n, W):
    """Oracle DFS walk; per cell-visit record true pos and the (2W+1)^2 local wall+landmark submap (allo)."""
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    wall, col = env.wall, col_norm(env.col); Wg = wall.shape[0]
    T, C, D = [], [], []
    last = None

    def rec():
        cx, cy = int(env.px), int(env.py)
        patch = np.zeros((2 * W + 1, 2 * W + 1, 2), np.float32)
        for a in range(-W, W + 1):
            for b in range(-W, W + 1):
                yy, xx = cy + b, cx + a
                if 0 <= yy < Wg and 0 <= xx < Wg:
                    patch[b + W, a + W, 0] = wall[yy, xx]
                    patch[b + W, a + W, 1] = col[yy, xx]
        T.append(np.array([env.px, env.py], float)); C.append(cy * 100000 + cx); D.append(patch.ravel())

    rec(); last = (int(env.px), int(env.py))
    for (cx, cy) in walk[1:]:
        wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(40):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            err = wrap(math.atan2(dy, dx) - env.ang)
            if abs(err) > 0.20: a = 3 if err > 0 else 2
            else: a = 0
            motion(env, a)
            cell = (int(env.px), int(env.py))
            if cell != last:
                rec(); last = cell
    D = np.stack(D); nrm = np.linalg.norm(D, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return np.array(T), np.array(C), (D / nrm).astype(np.float32)


def col_norm(col):
    c = col.astype(np.float32).copy()
    m = c.max()
    return c / m if m > 0 else c


def auc(true, cell, desc, n_pairs=6000, seed=0):
    rng = np.random.default_rng(seed); N = len(cell)
    g = defaultdict(list)
    for i in range(N): g[cell[i]].append(i)
    rev = [v for v in g.values() if len(v) >= 2]
    same, diff = [], []
    while len(same) < n_pairs and rev:
        gg = rev[rng.integers(len(rev))]; i, j = rng.choice(len(gg), 2, replace=False)
        same.append(float(desc[gg[i]] @ desc[gg[j]]))
    while len(diff) < len(same):
        a, b = rng.integers(N), rng.integers(N)
        if cell[a] != cell[b]: diff.append(float(desc[a] @ desc[b]))
    same, diff = np.array(same), np.array(diff)
    allv = np.concatenate([same, diff]); order = allv.argsort()
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(allv))
    ns, nd = len(same), len(diff)
    return (ranks[:ns].sum() - ns * (ns - 1) / 2) / (ns * nd)


def gated_top1(true, cell, desc, R=4.0, seed=0):
    """For each node that revisits an earlier cell, among all earlier nodes within metric radius R of its
    TRUE position, is the highest-descriptor-sim one actually the same cell? (oracle-gate top-1 precision)."""
    N = len(cell); hits = tot = 0
    first_seen = {}
    inv = 1.0 / R; buckets = defaultdict(list); key = np.floor(true * inv).astype(int)
    for i in range(N):
        bx, by = key[i]
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += buckets.get((bx + dx, by + dy), [])
        cand = [j for j in cand if j < i - 2 and np.sum((true[i] - true[j]) ** 2) < R * R]
        if cell[i] in [cell[j] for j in cand] and cand:           # a true loop-closure is available in the gate
            tot += 1
            j_best = max(cand, key=lambda j: float(desc[i] @ desc[j]))
            if cell[j_best] == cell[i]: hits += 1
        buckets[(bx, by)].append(i)
    return hits / max(tot, 1), tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--Ws", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--R", type=float, default=4.0); ap.add_argument("--mazes", type=int, default=3)
    ap.add_argument("--ambiguity", type=int, default=1); ap.add_argument("--lm_density", type=float, default=0.15)
    a = ap.parse_args()
    print(f"ORACLE local-submap descriptor: AUC and gated top-1 precision (R={a.R}) vs size and radius W "
          f"[ambiguity={a.ambiguity} lm_density={a.lm_density}]")
    print(f"{'n':>4} {'grid':>8} {'cells':>6} | " + " ".join(f"W{w}:AUC  W{w}:top1 " for w in a.Ws))
    for n in a.sizes:
        segs = []; ncell = None
        for W in a.Ws:
            ts, cs, ds = [], [], []
            for m in range(a.mazes):
                env = TrackMazeEnv(n=n, ambiguity=a.ambiguity, lm_density=a.lm_density, loop=0.3, seed=9000 + m, max_steps=10 ** 9)
                env.reset(); t, c, d = rollout_submap(env, n, W); ts.append(t); cs.append(c); ds.append(d)
            t = np.concatenate(ts); c = np.concatenate(cs); d = np.concatenate(ds)
            ncell = len(np.unique(c))
            au = auc(t, c, d); p1, tot = gated_top1(t, c, d, a.R)
            segs.append(f"{au:6.3f} {p1:6.3f}")
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} {ncell:>6} | " + "  ".join(segs))
