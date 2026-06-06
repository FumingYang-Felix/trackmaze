"""Round 5 (capstone): replace R3-B's ORACLE place-identity with a LEARNED, metric-GATED loop closure and
test end-to-end global tracking vs size, trained small. This is the real bet: can learned recognition +
metric gating recover the oracle's size-invariant global bound?

Pipeline (oracle-heading odometry regime to isolate the RECOGNITION question -- heading correction is the
separate R2/SE(2) piece): drive the oracle DFS walk; per cell-visit record (true pos, drifting odometry est,
raw omni view, cell). A rotation-invariant encoder (trained on n=6,12 only) gives a heading-free place
descriptor. Loop-closure candidates = node pairs whose ESTIMATES are within a drift-scaled radius R (metric
gate) AND whose descriptors match (sim > thr). Accepted pairs become loop edges in the R3-B pose-graph.

Reports, per size: loop-closure precision/recall (vs true same-cell), and global mean error for
  dead   : no loop closure (dead-reckon pose-graph)        -- grows
  learned: learned + metric-gated loop closure             -- the method
  oracle : oracle place-identity loop closure (R3-B)        -- upper bound
"""
import argparse, math, sys, numpy as np, torch
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, dfs_walk, wrap, motion
from round3b import solve_pose_graph, pick_keep
from round4b import RotInv, feats, build_pairs, train_encoder, gen_dfs

sys.setrecursionlimit(2_000_000)


def rollout(env, n):
    """Oracle DFS walk; per cell-visit record true disp, oracle-heading odometry est, raw omni view, cell."""
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    true0 = np.array([env.px, env.py], float); est = np.zeros(2)
    T, E, C, L = [], [], [], []
    last = None

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
                a = 0; est = est + np.array([math.cos(env.ang), math.sin(env.ang)]) * 0.22  # oracle-heading odo
            motion(env, a)
            cell = (int(env.px), int(env.py))
            if cell != last:
                rec(); last = cell
    return (np.array(T), np.array(E), np.array(C), np.stack(L).astype(np.float32))


def gated_loops(est, desc, R, thr):
    """Candidate loop pairs via a metric gate (grid-bucketed est within radius R) then descriptor match."""
    N = len(est); loops = []
    inv = 1.0 / R; buckets = defaultdict(list)
    key = np.floor(est * inv).astype(int)
    for i in range(N): buckets[(key[i, 0], key[i, 1])].append(i)
    R2 = R * R
    for i in range(N):
        bx, by = key[i]
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += buckets.get((bx + dx, by + dy), [])
        for j in cand:
            if j <= i + 2: continue                                   # skip near-consecutive (already odo-linked)
            d = est[i] - est[j]
            if d[0] * d[0] + d[1] * d[1] < R2 and float(desc[i] @ desc[j]) > thr:
                loops.append((i, j))
    return loops


def lc_quality(loops, cell):
    if not loops: return 0.0, 0.0
    tp = sum(1 for i, j in loops if cell[i] == cell[j])
    # recall vs reachable same-cell pairs within the same rollout is hard to count exactly; report match-rate
    return tp / len(loops), len(loops)


def run_size(n, enc, R, thr, n_mazes=8, seed0=9000):
    res = {"dead": [], "learned": [], "oracle": []}; prec = []; nl = []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset()
        true, est, cell, view = rollout(env, n)
        with torch.no_grad(): desc = enc(torch.tensor(view)).numpy()
        # oracle loops
        first = {}; oloops = []
        for i, c in enumerate(cell):
            if c in first: oloops.append((first[c], i))
            else: first[c] = i
        lloops = gated_loops(est, desc, R, thr)
        p, k = lc_quality(lloops, cell); prec.append(p); nl.append(k)
        N = len(true); keepall = set(range(N))
        res["dead"].append(solve_pose_graph(true, est, [], keepall))
        res["learned"].append(solve_pose_graph(true, est, lloops, keepall))
        res["oracle"].append(solve_pose_graph(true, est, oloops, keepall))
    return {k: float(np.mean(v)) for k, v in res.items()}, float(np.mean(prec)), float(np.mean(nl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=4.0); ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--mazes", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    a = ap.parse_args()
    print(f"Round 5: learned metric-gated loop closure, train encoder on n=6,12, eval OOD. R={a.R} thr={a.thr}")
    tr6, tr12 = gen_dfs(6, 4, "true", 7000), gen_dfs(12, 3, "true", 7100)
    tr = {k: np.concatenate([tr6[k], tr12[k]]) for k in ("geo", "lm", "cell")}
    tr["ep"] = np.concatenate([tr6["ep"], 1000 + tr12["ep"]])
    enc = train_encoder(RotInv(), feats(tr), build_pairs(tr), epochs=a.epochs)
    print(f"  encoder trained ({len(build_pairs(tr))} revisited cells)\n")
    print(f"{'n':>4} {'grid':>8} | {'LC_prec':>7} {'LC_n':>6} | {'dead':>7} {'learned':>8} {'oracle':>7}")
    g = {"dead": [], "learned": [], "oracle": []}
    for n in a.sizes:
        r, p, k = run_size(n, enc, a.R, a.thr, n_mazes=a.mazes)
        for kk in g: g[kk].append(r[kk])
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {p:7.3f} {k:6.0f} | {r['dead']:7.2f} {r['learned']:8.2f} {r['oracle']:7.2f}")
    print("\nGROWTH (n_min -> n_max):")
    for kk in g:
        print(f"  {kk:>8}: {g[kk][0]:.2f} -> {g[kk][-1]:.2f}  ({g[kk][-1]-g[kk][0]:+.2f})")
