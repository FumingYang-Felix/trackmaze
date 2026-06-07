"""b04: was the "61x61 winding wall" FUNDAMENTAL, or an artifact of my closure TOPOLOGY?

b03e closed every later visit to the FIRST visit of a cell -> a closure edge spans the ENTIRE first->last
time gap. The heading random walk (rot_noise=0.09) accumulates drift std = 0.09*sqrt(span). For a span of
~T=40n steps that exceeds pi (winding-ambiguous) around n~30 (61x61). THAT is the "wall" -- but it is a
property of the EDGE SPAN, not of the problem. Pose-graph SLAM theory: absolute error at a node is bounded by
the graph distance (effective resistance) to the anchor, NOT by total path length, IF the loop-closure graph
is well connected with LOW-DRIFT edges.

Fix tested here: close each revisit to its MOST RECENT prior visit (minimal edge span -> minimal per-edge
drift -> winding-safe), and add robust (DCS) down-weighting for the ~5% wrong matcher closures. If heading
error now stays FLAT to 64/89/129 (vs b03e blowing up at ~61), the wall was my topology -> heading is NOT
fundamentally bounded; it is a data-association/closure-graph problem (the real frontier).

Compares: cmd-only (open loop) | b03e-style all-to-first | b04 most-recent+robust, abs heading err (deg) vs size.
"""
import sys, os, math, numpy as np
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
    rec = []
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


def _q90(x):
    return x - round(x / HALF) * HALF


def build_closures(rec, matcher, topology, oracle=False):
    """topology='first': every later visit -> first visit (b03e). 'recent': each visit -> most recent prior.
    oracle=True: use TRUE relative heading at closures (perfect place-id + perfect quadrant) -> isolates whether
    the limit is the MATCHER (engineerable) or GAUGE (fundamental)."""
    turns = np.array([r[1] for r in rec]); cmd = np.cumsum(turns)
    closures = []; bycell = defaultdict(list)
    for t, r in enumerate(rec): bycell[r[2]].append(t)
    for c, ts in bycell.items():
        if len(ts) < 2: continue
        if topology == 'first':
            pairs = [(ts[0], b) for b in ts[1:]]
        else:  # 'recent'
            pairs = [(ts[k - 1], ts[k]) for k in range(1, len(ts))]
        for (a, b) in pairs:
            if oracle:
                rel = rec[a][3] - rec[b][3]   # TRUE relative heading-from-start (continuous, winding-correct)
            else:
                rq = matcher_relquad(matcher, rec[a][4], rec[a][5], rec[b][4], rec[b][5])
                mrel = wrap(wrap(rec[a][0] - rec[b][0]) + rq * HALF)
                rel = (cmd[a] - cmd[b]) + wrap(mrel - (cmd[a] - cmd[b]))
            closures.append((a, b, rel))   # (i=a, j=b): h_a - h_b ~ rel
    return closures, cmd


def solve_heading(rec, closures, cmd, w_close=40.0, w_grid=8.0, iters=15, robust=True, dcs_delta=0.35):
    T = len(rec)
    turns = np.array([r[1] for r in rec])
    grid0 = rec[0][0]; base = np.array([wrap(r[0] - grid0) for r in rec])
    h = cmd.copy()
    for _ in range(iters):
        rows, cols, vals, rhs = [], [], [], []; nE = [0]
        def con(i, j, tgt, w):
            r = wrap((h[i] - h[j]) - tgt)
            rows.extend([nE[0], nE[0]]); cols.extend([i, j]); vals.extend([w, -w]); rhs.append(-w * r); nE[0] += 1
        for t in range(1, T): con(t, t - 1, turns[t], 1.0 / 0.09 ** 2)              # smoothness
        for (a, b, rel) in closures:                                               # matcher closures (robust)
            w = w_close
            if robust:
                r = wrap((h[a] - h[b]) - rel); w = w_close * (dcs_delta ** 2 / (dcs_delta ** 2 + r ** 2))  # DCS
            con(a, b, rel, w)
        for t in range(T):                                                         # grid mod-90 (drift-free fine)
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
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64, 89])
    ap.add_argument("--mazes", type=int, default=3); ap.add_argument("--wclose", type=float, default=40.0)
    a = ap.parse_args()
    print("Training matcher (n=6,12)..."); m = train_matcher(epochs=1200)
    print("b04: closure-topology + ORACLE-closure test. abs heading err (deg) vs size.")
    print("ORACLE col = perfect place-id+quadrant -> isolates GAUGE (fundamental) from MATCHER (engineerable).")
    print(f"{'n':>4} {'grid':>9} | {'cmd-only':>8} {'matchRecent':>11} {'ORACLErecent':>12} {'ORACLEfirst':>11}  (#edges)")
    for n in a.sizes:
        cm, mr, orc, orf, ne = [], [], [], [], []
        for mi in range(a.mazes):
            rec = drive(n, 5000 + mi)
            cl_r, cmd = build_closures(rec, m, 'recent')
            cl_or, _ = build_closures(rec, m, 'recent', oracle=True)
            cl_of, _ = build_closures(rec, m, 'first', oracle=True)
            er, c = solve_heading(rec, cl_r, cmd, a.wclose, robust=True)
            eor, _ = solve_heading(rec, cl_or, cmd, 200.0, robust=False)
            eof, _ = solve_heading(rec, cl_of, cmd, 200.0, robust=False)
            cm.append(c); mr.append(er); orc.append(eor); orf.append(eof); ne.append(len(cl_r))
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {np.mean(cm):8.1f} {np.mean(mr):11.1f} {np.mean(orc):12.1f} {np.mean(orf):11.1f}  ({int(np.mean(ne))})")
    print("\nORACLE flat with size => limit is the MATCHER/place-recognition (engineerable), NOT gauge.")
    print("ORACLE grows with size => absolute heading is GAUGE-fundamental even with perfect closures.")
