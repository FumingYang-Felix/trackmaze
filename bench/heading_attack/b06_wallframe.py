"""b06: THE REFRAME. Track heading relative to the WALLS, not the START.

b05 proved: heading-relative-to-START is gauge-bounded (single anchor -> error ~ graph-distance ~ size). But
that is the WRONG frame. The maze WALLS define a global frame (4 cardinal directions). The grid (freq-4 Fourier
of the omni view) reads the agent's heading mod 90 RELATIVE TO THE WALLS, drift-free, size-invariantly. So:

   heading_wall = grid_fine (mod 90, drift-free, ~size-invariant noise)  +  90 * k   (integer quadrant)

k is an INTEGER that changes by +-1 only when the agent's heading crosses a 90-deg wall boundary. That is a
RARE, LOCAL, drift-free EVENT: detect it from the grid-phase wrap + the (known) sign of the commanded turn.
Crucially this does NOT integrate drift (unlike cmd-from-start): k accrues no random walk, only occasional
crossing-detection mistakes (suppressed by hysteresis, and fixable by loop closure). The only irreducible
unknown is k_0 (which wall direction is 'north') -- a SINGLE global 2-bit constant from the 4-fold symmetry,
CONSTANT with size, not growing.

If heading_wall error is FLAT with size (~grid noise) => absolute heading IS size-invariant in the wall frame
=> my earlier 'gauge-bounded' verdict was a FRAME error, and global state-tracking is recoverable. Decisive.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
HALF = math.pi / 2


def _q90dist(d):  # circular distance in mod-90 space (rad)
    d = d % HALF
    return min(d, HALF - d)


def run_size(n, n_mazes, seed0, dead=0.30):
    """GRID-ANCHORED integer tracker: predict heading from local command (h_pred = h_prev + turn), snap the
    integer quadrant to the DRIFT-FREE grid every step (k = round((h_pred - gh)/HALF); h_est = gh + HALF*k).
    Nothing accumulates -> no random walk; only occasional integer-pick mistakes. Report abs heading err (deg),
    grid mod-90 floor (offset-removed), cmd-from-start (drift baseline), and the k-tracking error fraction."""
    e_wall, e_gridfloor, e_cmd, k_wrong = [], [], [], []
    for mi in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + mi, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0; T = 40 * n
        cmd = 0.0; h_est = None; gh0 = None; prev_turn = 0.0
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                e = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
                turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
                g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
                gh = -_grid_heading(g)                        # NEGATE: _grid_heading is reflected (gh~HALF-ang%H);
                #                                               co-rotate with the command so prediction aligns
                if h_est is None:
                    h_est = gh; gh0 = gh                       # anchor
                else:
                    h_pred = h_est + prev_turn                 # predict CURRENT from prev step's applied turn
                    k = round((h_pred - gh) / HALF)            # integer quadrant, snapped to grid
                    h_est = gh + HALF * k                      # correct to drift-free grid -> no accumulation
                true = wrap(env.ang - ang0)
                est_rel = wrap(h_est - gh0)                    # heading rel start via wall-frame tracking
                e_wall.append(abs(wrap(est_rel - true)))
                e_gridfloor.append(_q90dist(gh - env.ang))     # offset~0; the mod-90 floor
                e_cmd.append(abs(wrap(cmd - true)))
                k_true = round((true - wrap(gh - gh0)) / HALF)
                k_est = round((est_rel - wrap(gh - gh0)) / HALF)
                k_wrong.append(1.0 if (k_true % 4) != (k_est % 4) else 0.0)
                if a == 2: cmd -= 0.20
                elif a == 3: cmd += 0.20
                prev_turn = turn
                motion(env, a); steps += 1
    D = lambda x: np.mean(x) * 180 / math.pi
    return D(e_wall), D(e_gridfloor), D(e_cmd), float(np.mean(k_wrong))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64, 89])
    ap.add_argument("--mazes", type=int, default=3); ap.add_argument("--dead", type=float, default=0.30)
    a = ap.parse_args()
    print("b06: heading in the WALL frame (grid-anchored integer tracker). abs err (deg) vs size.")
    print(f"{'n':>4} {'grid':>9} | {'wall-frame':>10} {'grid floor':>10} {'cmd-start':>10} {'k-wrong%':>9}")
    for n in a.sizes:
        ew, eg, ec, kw = run_size(n, a.mazes, 5000, a.dead)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {ew:10.1f} {eg:10.1f} {ec:10.1f} {kw*100:8.1f}%")
    print("\nwall-frame ~= grid floor & FLAT with size => absolute heading is SIZE-INVARIANT in the wall frame.")
    print("=> earlier 'gauge-bounded' was a FRAME error; only k0 (one global 2-bit, from 4-fold symmetry) is unknown.")
