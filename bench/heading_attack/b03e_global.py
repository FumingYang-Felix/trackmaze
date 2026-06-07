"""b03e (branch A): GLOBAL batch optimization of absolute heading, using the 95% matcher closures.

Greedy online fusion failed (b03b/d) because the quadrant correction doesn't persist against the constant
cmd/grid pull. The principled fix: batch optimization. Decompose h_t = base_t + q_t*90 where
  base_t = wrap(grid_t - grid0)  is the DRIFT-FREE mod-90 part (grid, anchored to the start reading),
  q_t (integer) is the QUADRANT. Solve for q_t with:
   - smoothness (chain): q_t - q_{t-1} ~ (cmd_turn_t - (base_t - base_{t-1}))/90      [soft, drift slack]
   - closures: for same-cell visits i,j, q_i - q_j ~ (matcher_full_rel - (base_i-base_j))/90  [95%, strong]
   - anchor: q_0 = 0.
1D weighted least-squares on the (chain + loop-closure) graph -> continuous q -> round -> h = base + q*90.
This is the heading analogue of pose-graph SLAM; it distributes the 95% closure evidence globally instead of
greedily. If it resolves heading (error << b02's 67deg, flat with size) -> heading SOLVED (with place-id) ->
Stage-1 unblocked. If even global-opt drifts -> heading is gauge-bounded like position.
"""
import sys, os, math, numpy as np, torch
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
from heading_attack.b03d_integrated import train_matcher, matcher_relquad
HALF = math.pi / 2


def drive(n, seed):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; cmd = 0.0
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0; T = 40 * n
    rec = []  # per step: (gh, cmd_turn, cell, true_rel, g, l)
    for (cx, cy) in walk[1:]:
        if steps >= T: break
        wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(60):
            if steps >= T: break
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            rec.append((_grid_heading(g), turn, (int(env.px), int(env.py)), wrap(env.ang - ang0), g, l))
            if a == 2: cmd -= 0.20
            elif a == 3: cmd += 0.20
            motion(env, a); steps += 1
    return rec


def _q90(x):  # wrap to (-pi/4, pi/4]  (the mod-90 fine residual, quadrant-preserving)
    return x - round(x / HALF) * HALF


def solve_heading(rec, matcher, w_close=200.0, w_grid=8.0, iters=12):
    """ANGULAR pose-graph (Gauss-Newton, wrapped residuals; CORRECT GN sign rhs=-w*r). Estimates continuous
    heading h_t. Constraints: smoothness (h_t-h_{t-1}=cmd turn), grid per-step (h_t mod 90 = grid, drift-free
    fine), 95%-matcher closures (relative heading at same-cell revisits), anchor (h_0=0)."""
    T = len(rec)
    turns = np.array([r[1] for r in rec]); cmd = np.cumsum(turns)
    grid0 = rec[0][0]; base = np.array([wrap(r[0] - grid0) for r in rec])
    h = cmd.copy()
    # precompute closure edges (matcher). WINDING-correct: rel = cmd-relative (winding) + wrapped drift-diff.
    closures = []
    bycell = defaultdict(list)
    for t, r in enumerate(rec): bycell[r[2]].append(t)
    for c, ts in bycell.items():
        a = ts[0]                                                # FIRST visit (lowest drift) as reference
        for b in ts[1:]:                                         # close every later visit to it (LONG-RANGE)
            rq = matcher_relquad(matcher, rec[a][4], rec[a][5], rec[b][4], rec[b][5])
            mrel = wrap(wrap(rec[a][0] - rec[b][0]) + rq * HALF)  # matcher relative heading (mod 2pi)
            rel = (cmd[a] - cmd[b]) + wrap(mrel - (cmd[a] - cmd[b]))   # winding from cmd + small drift correction
            closures.append((a, b, rel))
    for _ in range(iters):
        rows, cols, vals, rhs = [], [], [], []; nE = [0]
        def con(i, j, tgt, w):       # (h_i - h_j) ~ tgt
            r = wrap((h[i] - h[j]) - tgt)
            rows.extend([nE[0], nE[0]]); cols.extend([i, j]); vals.extend([w, -w]); rhs.append(-w * r); nE[0] += 1
        for t in range(1, T): con(t, t - 1, turns[t], 1.0 / 0.09 ** 2)        # smoothness
        for (a, b, rel) in closures: con(a, b, rel, w_close)                  # matcher closures
        for t in range(T):                                                   # grid mod-90 (fine, drift-free)
            rg = _q90(h[t] - base[t]); rows.append(nE[0]); cols.append(t); vals.append(w_grid); rhs.append(-w_grid * rg); nE[0] += 1
        ra = wrap(h[0] - 0.0); rows.append(nE[0]); cols.append(0); vals.append(1e4); rhs.append(-1e4 * ra); nE[0] += 1  # anchor
        A = sp.csr_matrix((vals, (rows, cols)), shape=(nE[0], T))
        h = h + spla.spsolve((A.T @ A).tocsc() + 1e-9 * sp.eye(T), A.T @ np.array(rhs))
    true_rel = np.array([r[3] for r in rec])
    err = np.mean([abs(wrap(h[t] - true_rel[t])) for t in range(T)]) * 180 / math.pi
    cmd_err = np.mean([abs(wrap(cmd[t] - true_rel[t])) for t in range(T)]) * 180 / math.pi
    return err, cmd_err


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--mazes", type=int, default=4); ap.add_argument("--wclose", type=float, default=20.0)
    a = ap.parse_args()
    print("Training matcher..."); m = train_matcher(epochs=1200)
    print(f"b03e GLOBAL batch heading optimization (grid mod-90 + 95% matcher closures + cmd smoothness + anchor)")
    print(f"{'n':>4} {'grid':>8} | {'b03e global':>11} {'cmd-only':>8}  (deg; vs b02 learned 30->67)")
    for n in a.sizes:
        es, cs = [], []
        for mi in range(a.mazes):
            r = drive(n, 5000 + mi); e, c = solve_heading(r, m, a.wclose); es.append(e); cs.append(c)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {np.mean(es):11.1f} {np.mean(cs):8.1f}")
    print("\nb03e << 67 & flat => global-opt resolves the quadrant => heading SOLVED (place-id). else gauge-bounded.")
