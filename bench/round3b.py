"""Round 3-B (training-free mechanism test): GLOBAL consistency via BATCH POSE-GRAPH least-squares, and
whether a BOUNDED, importance-weighted pose graph (compression) matches the FULL one.

3-A showed greedy snap-to-anchor only re-zeros error relative to the start; MEAN error over fresh territory
still grows for ALL memory schemes (even unbounded). The right global estimator is pose-graph optimization:
nodes = cell-visits, ODOMETRY edges (relative dead-reckon delta between consecutive nodes), LOOP-CLOSURE
edges (revisited cell => the two visits are the SAME true place => relative position 0). Jointly minimizing
the weighted residuals distributes loop-closure corrections across the whole graph (not just near start).

Translation-only + isotropic weights => x and y decouple into two weighted-Laplacian linear solves.

Compression: keep only K of the N nodes; marginalize the rest (odometry telescopes: relative dead-reckon
between kept nodes a<b is just est[b]-est[a]). Importance variants for WHICH K to keep:
  full      : keep all N nodes (unbounded; O(n^2))
  loops K   : keep every loop-closure endpoint first, then fill to K with uniform spread  (the hypothesis)
  fifo  K   : keep the most RECENT K nodes (drops old loop endpoints -> loses constraints)
  spread K  : keep every (N/K)-th node uniformly (ignores loop structure)

Decisive read: if `loops K` (K << N) matches `full` mean-error-vs-size while `fifo` degrades, then
importance-weighted compression of the pose graph preserves global consistency with sublinear memory.
The residual full-graph growth (if any) is the FUNDAMENTAL single-gauge-anchor SLAM limit (error grows with
constraint-distance to the one fixed anchor), NOT a memory limit -- infinite memory can't beat it either.
"""
import argparse, math, sys, numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from env import TrackMazeEnv
from round3a import cell_graph, dfs_walk, wrap, motion

sys.setrecursionlimit(2_000_000)


def explore_and_log(env, n, oracle_heading=True):
    """Drive the oracle DFS walk; emit pose-graph nodes (true pos, dead-reckon est, cell) + loop edges.
    oracle_heading=True isolates the TRANSLATION/memory question (heading taken from truth, so the only
    drift is translation odometry noise + collisions -- translation-only pose-graph is then the correct
    estimator). oracle_heading=False integrates commanded heading too -> heading random-walk dominates."""
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    true0 = np.array([env.px, env.py], dtype=np.float64)
    est = np.zeros(2); ang_est = env.ang
    nodes = []; cell_first = {}; loops = []; last_cell = None

    def add_node():
        cell = (int(env.px), int(env.py))
        nodes.append((np.array([env.px, env.py], dtype=np.float64) - true0, est.copy(), cell))
        idx = len(nodes) - 1
        if cell in cell_first: loops.append((cell_first[cell], idx))   # same true place -> rel position 0
        else: cell_first[cell] = idx
        return cell

    last_cell = add_node()
    for (cx, cy) in walk[1:]:
        wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(40):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            err = wrap(math.atan2(dy, dx) - env.ang)
            if abs(err) > 0.20:
                if err > 0: ang_est += 0.20; a = 3
                else:       ang_est -= 0.20; a = 2
            else:
                a = 0
                h = env.ang if oracle_heading else ang_est
                est = est + np.array([math.cos(h), math.sin(h)]) * 0.22
            motion(env, a)
            cell = (int(env.px), int(env.py))
            if cell != last_cell:
                last_cell = add_node()
    return nodes, loops


def solve_pose_graph(true_xy, est_xy, loops, keep, w_loop=5.0):
    """Least-squares over kept nodes: odometry edges (consecutive kept, telescoped est) + loop edges.
    Then RECONSTRUCT every original node (kept anchor's solved pos + dead-reckon from that anchor) and
    return mean ||recon - true|| over ALL nodes -- a fair apples-to-apples eval set for every variant.
    Node 0 (start) is the fixed gauge anchor."""
    keep = sorted(keep)
    pos = {idx: i for i, idx in enumerate(keep)}
    N = len(keep); Ntot = len(true_xy)
    edges = []
    for i in range(N - 1):
        a, b = keep[i], keep[i + 1]
        edges.append((i, i + 1, est_xy[b] - est_xy[a], 1.0))               # telescoped odometry
    for (i, j) in loops:
        if i in pos and j in pos:
            edges.append((pos[i], pos[j], np.zeros(2), w_loop))            # same place -> rel 0

    L = sp.lil_matrix((N, N)); bx = np.zeros(N); by = np.zeros(N)
    for a, b, d, w in edges:
        L[a, a] += w; L[b, b] += w; L[a, b] -= w; L[b, a] -= w
        bx[a] -= w * d[0]; bx[b] += w * d[0]; by[a] -= w * d[1]; by[b] += w * d[1]
    anchor = pos[0]
    free = [i for i in range(N) if i != anchor]
    Lf = L.tocsr()[free][:, free]
    xs = np.zeros(N); ys = np.zeros(N)
    xs[free] = spla.spsolve(Lf, bx[free]); ys[free] = spla.spsolve(Lf, by[free])
    solved = np.stack([xs, ys], 1)                                        # kept-node corrected positions

    # reconstruct ALL nodes: localize between kept anchors by dead-reckon from the most recent kept anchor
    keep_arr = np.array(keep); recon = np.zeros((Ntot, 2)); ki = 0
    for idx in range(Ntot):
        while ki + 1 < N and keep_arr[ki + 1] <= idx: ki += 1
        a = keep_arr[ki]
        recon[idx] = solved[ki] + (est_xy[idx] - est_xy[a])
    true_all = np.stack(true_xy)
    return float(np.mean(np.linalg.norm(recon - true_all, axis=1)))


def pick_keep(N, loops, variant, K):
    if variant == "full" or K >= N: return set(range(N))
    if variant == "fifo":   keep = set(range(N - K, N))
    elif variant == "spread":
        keep = set(np.linspace(0, N - 1, K).round().astype(int).tolist())
    else:  # loops: loop endpoints first, then uniform spread to K
        keep = set()
        for i, j in loops: keep.add(i); keep.add(j)
        if len(keep) > K:
            keep = set(sorted(keep)[:: max(1, len(keep) // K)])
        for idx in np.linspace(0, N - 1, K).round().astype(int):
            if len(keep) >= K: break
            keep.add(int(idx))
    keep.add(0); keep.add(N - 1)
    return keep


def run(n, variant, K, n_mazes=12, oracle_heading=True):
    errs, Ns, kept = [], [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=3000 + m, max_steps=10 ** 9)
        env.reset()
        nodes, loops = explore_and_log(env, n, oracle_heading=oracle_heading)
        true_xy = [nd[0] for nd in nodes]; est_xy = [nd[1] for nd in nodes]; N = len(nodes)
        keep = pick_keep(N, loops, variant, K)
        errs.append(solve_pose_graph(true_xy, est_xy, loops, keep)); Ns.append(N); kept.append(len(keep))
    return float(np.mean(errs)), float(np.mean(Ns)), float(np.mean(kept))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=80)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=12)
    ap.add_argument("--cmd_heading", action="store_true", help="integrate commanded heading (heading drifts) instead of oracle heading")
    a = ap.parse_args()
    oh = not a.cmd_heading
    variants = [("full", 0), ("loops", a.K), ("spread", a.K), ("fifo", a.K)]
    print(f"BATCH POSE-GRAPH mean node error vs size (K={a.K}, {a.mazes} mazes, loop=0.3, "
          f"{'ORACLE' if oh else 'CMD'} heading, oracle DFS+place-id)")
    print(f"{'n':>4} {'grid':>8} " + " ".join(f"{v[0]:>9}" for v in variants) + f" | {'nodes':>6} {'keptK':>6}")
    grow = {v[0]: [] for v in variants}
    for n in a.sizes:
        row = []; Nn = kk = 0
        for name, K in variants:
            e, N, kp = run(n, name, K, n_mazes=a.mazes, oracle_heading=oh); row.append(e); grow[name].append(e)
            if name == "full": Nn = N
            if name == "loops": kk = kp
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} " + " ".join(f"{e:9.2f}" for e in row) + f" | {Nn:6.0f} {kk:6.0f}")
    print("\nGROWTH of mean error (n_min -> n_max):")
    for name, _ in variants:
        g = grow[name]; print(f"  {name:>9}: {g[0]:.2f} -> {g[-1]:.2f}  ({g[-1]-g[0]:+.2f})")
    g = grow["full"]
    print(f"\nfull-graph growth ratio (n{a.sizes[-1]}/n{a.sizes[0]}) = {g[-1]/max(g[0],1e-9):.2f}x "
          f"(sqrt(size ratio) = {math.sqrt(a.sizes[-1]/a.sizes[0]):.2f}x; linear = {a.sizes[-1]/a.sizes[0]:.2f}x)")
