"""b05: ESTIMATOR-FREE information floor for absolute heading. Answers the user's question cleanly:
is absolute heading GAUGE-bounded (grows with size), or recoverable (flat) -- given the loop-closure graph?

Heading is a Gaussian MRF: each step adds drift ~ N(0, s^2), s=rot_noise=0.09 (chain edges, conductance 1/s^2).
Each loop closure (revisit, place-id known) measures relative heading with noise sc (closure edges, conductance
1/sc^2). Anchor = node 0 (h_0=0, grounded). The MLE/BLUE variance of absolute heading at node t equals the
EFFECTIVE RESISTANCE R_eff(t,0) in this conductance graph. std_t = sqrt(R_eff(t,0)). This is the floor NO
estimator can beat. We compute it exactly (grounded-Laplacian inverse diagonal). No matcher, no least-squares
tuning -- pure information.

Two regimes:
  EXPLORE (DFS, each cell ~1-2 visits): sparse closures -> is the floor flat or growing with size?
  REVISIT-DENSE (re-walk corridors K times): more closures -> does the floor flatten? (the cost of bounded global)

If EXPLORE floor grows with size but DENSE floor stays flat => absolute heading is recoverable at ANY size, at
the COST of revisit density (loop closures). NOT fundamentally bounded -- it's a coverage/data-association
trade-off. If even DENSE grows => gauge-fundamental.
"""
import sys, os, math, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from round3a import cell_graph, dfs_walk, wrap, motion


def drive_cells(n, seed, revisit=1):
    """Return per-step cell list along a DFS walk. revisit>1 re-walks the full DFS order that many times
    (denser loop closures). T auto = len(walk)*revisit (no truncation -> full coverage)."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed, max_steps=10 ** 9); env.reset()
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    cells = []
    for rep in range(revisit):
        order = walk if rep % 2 == 0 else walk[::-1]
        for (cx, cy) in order:
            cells.append((cx, cy))
    return cells


def heading_floor(cells, s=0.09, sc=0.10):
    """std of absolute-heading BLUE at each step = sqrt(eff. resistance to node 0). Edges: chain (var s^2),
    closures between consecutive same-cell visits (var sc^2). Returns mean & max std (deg) and #closures."""
    T = len(cells)
    rows, cols, w = [], [], []
    cc = 1.0 / s ** 2
    for t in range(1, T):
        rows += [t, t - 1]; cols += [t - 1, t]; w += [cc, cc]      # chain (symmetric)
    bycell = defaultdict(list)
    for t, c in enumerate(cells): bycell[c].append(t)
    nclos = 0; ccl = 1.0 / sc ** 2
    for c, ts in bycell.items():
        for k in range(1, len(ts)):
            a, b = ts[k - 1], ts[k]
            rows += [a, b]; cols += [b, a]; w += [ccl, ccl]; nclos += 1   # closure (consecutive revisits)
    W = sp.csr_matrix((w, (rows, cols)), shape=(T, T))
    deg = np.array(W.sum(1)).ravel()
    L = sp.diags(deg) - W
    # ground node 0: remove row/col 0, invert -> diag gives R_eff(t,0)
    keep = np.arange(1, T)
    Lg = L[keep][:, keep].tocsc()
    # R_eff(t,0) = (Lg^{-1})_tt ; get diagonal via solving for identity columns in blocks (T can be ~2.5k)
    n_ = Lg.shape[0]
    Reff = np.zeros(n_)
    I = sp.eye(n_, format='csc')
    # solve Lg X = I  (dense-ish but n_<=~2.5k ok); use splu once
    lu = spla.splu(Lg)
    for j in range(n_):
        e = np.zeros(n_); e[j] = 1.0
        Reff[j] = lu.solve(e)[j]
    std = np.sqrt(np.maximum(Reff, 0)) * 180 / math.pi
    return std.mean(), std.max(), T, nclos


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--mazes", type=int, default=2)
    ap.add_argument("--sc", type=float, default=0.10)   # closure measurement noise (rad); matcher fine ~0.10-0.35
    a = ap.parse_args()
    print("b05: information floor for ABSOLUTE heading = sqrt(eff.resistance to anchor). Estimator-free.")
    print(f"closure noise sc={a.sc} rad. drift s=0.09/step. mean(max) absolute-heading std floor (deg) vs size.")
    print(f"{'n':>4} {'grid':>9} | {'EXPLORE(1x)':>20} {'DENSE(3x)':>20}")
    print(f"{'':>4} {'':>9} | {'mean   max   #clo':>20} {'mean   max   #clo':>20}")
    for n in a.sizes:
        ex_m, ex_x, ex_c, dn_m, dn_x, dn_c = [], [], [], [], [], []
        for mi in range(a.mazes):
            c1 = drive_cells(n, 5000 + mi, revisit=1)
            m1, x1, T1, k1 = heading_floor(c1, sc=a.sc)
            c3 = drive_cells(n, 5000 + mi, revisit=3)
            m3, x3, T3, k3 = heading_floor(c3, sc=a.sc)
            ex_m.append(m1); ex_x.append(x1); ex_c.append(k1); dn_m.append(m3); dn_x.append(x3); dn_c.append(k3)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {np.mean(ex_m):5.1f} {np.mean(ex_x):5.1f} {int(np.mean(ex_c)):6d}     "
              f"{np.mean(dn_m):5.1f} {np.mean(dn_x):5.1f} {int(np.mean(dn_c)):6d}")
    print("\nEXPLORE grows + DENSE flat => heading recoverable at ANY size via revisit density (NOT fundamental).")
    print("both grow => gauge-fundamental. This is the floor; the matcher only needs to approach it.")
