"""b15: the RIGOROUS resolution of 'can we break gauge'. Estimator-free, via the synchronization information
floor = effective resistance to the anchor in the measurement graph (BLUE variance of the global frame). The
key new lens: this is a RANDOM-WALK RESISTANCE problem, governed by the DIMENSIONALITY of the environment's
loop topology:
  - 1D / tree (a maze is topologically near-tree): R_eff(node) ~ distance  -> grows LINEARLY -> the global
    frame decorrelates over a finite length, INDEPENDENT of maze size -> NOT recoverable beyond that length.
  - 2D-connected (open arena, many loops): R_eff ~ log(distance) -> grows only logarithmically (continuous)
    and the DISCRETE Z4 field has true long-range order -> recoverable at ~any size.
So 'breaking gauge' is a phase transition in ENVIRONMENT TOPOLOGY, not a solver question. We measure R_eff
(=> heading-std floor) vs graph-distance-from-anchor, for increasing loop density (tree -> loopy). If the
growth slope drops sharply with loop density (linear -> sublinear), the dimensional crossover is demonstrated.

(The single global 2-bit offset remains irreducible regardless, b08 -- that is environment SYMMETRY, separate.)
"""
import sys, os, math, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict, deque
from env import TrackMazeEnv
from round3a import cell_graph, dfs_walk, motion, wrap


def cell_adjacency(n, seed, loop):
    """Return the maze CELL adjacency (includes loop edges). loop=0 -> perfect maze = spanning TREE (1D-like);
    higher loop -> extra edges -> 2D-grid-like. This is the graph over which the global frame must propagate."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    adj = cell_graph(env.wall, n)            # dict cell->list of open-adjacent cells
    cells = sorted(adj.keys()); idx = {c: i for i, c in enumerate(cells)}
    edges = set()
    for c, nbrs in adj.items():
        for d in nbrs:
            a, b = idx[c], idx[d]
            if a < b: edges.add((a, b))
    return len(cells), list(edges), idx.get((0, 0), 0)


def floor_vs_distance(N, edges, anchor, s=0.09):
    """R_eff to anchor on the cell graph (each edge var s^2). Per-cell std (deg) + hop-distance from anchor."""
    cc = 1.0 / s ** 2; rows, cols, w = [], [], []; adj = defaultdict(list)
    for (a, b) in edges:
        rows.extend([a, b]); cols.extend([b, a]); w.extend([cc, cc]); adj[a].append(b); adj[b].append(a)
    W = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    L = sp.diags(np.array(W.sum(1)).ravel()) - W
    keep = np.array([i for i in range(N) if i != anchor]); Lg = L[keep][:, keep].tocsc(); lu = spla.splu(Lg)
    Reff = np.zeros(len(keep))
    for j in range(len(keep)):
        e = np.zeros(len(keep)); e[j] = 1.0; Reff[j] = lu.solve(e)[j]
    std = np.zeros(N); std[keep] = np.sqrt(np.maximum(Reff, 0)) * 180 / math.pi
    dist = np.full(N, -1); dist[anchor] = 0; dq = deque([anchor])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0: dist[v] = dist[u] + 1; dq.append(v)
    return std, dist


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--mazes", type=int, default=2)
    a = ap.parse_args()
    print(f"b15: global-frame heading-std floor (deg) vs hop-distance from anchor, on the CELL graph, by loop", flush=True)
    print(f"density. n={a.n} ({2*a.n+1}px). std~sqrt(R_eff): linear-in-dist=1D/tree(unrecoverable), log=2D(recoverable).", flush=True)
    print(f"{'loop':>6} | {'std@d=10':>9} {'std@d=30':>9} {'std@d=60':>9} | {'lin-fit slope':>13} {'log-fit':>8} {'#edges':>8}", flush=True)
    for loop in [0.0, 0.1, 0.3, 0.6, 0.9]:
        S, Dd, ne = [], [], 0
        for mi in range(a.mazes):
            N, edges, anchor = cell_adjacency(a.n, 5000 + mi, loop)
            std, dist = floor_vs_distance(N, edges, anchor); S.append(std); Dd.append(dist); ne += len(edges)
        std = np.concatenate(S); dist = np.concatenate(Dd)
        def at(d):
            m = (dist >= d - 4) & (dist <= d + 4); return std[m].mean() if m.any() else float('nan')
        msk = dist > 2
        lin = np.polyfit(dist[msk], std[msk], 1)[0]
        logc = np.polyfit(np.log(dist[msk]), std[msk], 1)[0]   # coefficient if std ~ c*log(dist)
        print(f"{loop:>6.1f} | {at(10):9.1f} {at(30):9.1f} {at(60):9.1f} | {lin:13.3f} {logc:8.2f} {ne // a.mazes:8d}", flush=True)
    print("\nloop=0 (perfect maze=tree): std grows LINEARLY (lin-fit slope high) => 1D => finite decorrelation =>", flush=True)
    print("global frame NOT size-invariant. As loop rises: linear slope drops, log-fit dominates => 2D => recoverable.", flush=True)
