"""b07: THE POSITIVE RESULT. Heading = drift-free fine (grid, ~5deg) + integer quadrant. The quadrant is the
only thing that drifts (sqrt(steps) via turn-integration). RE-ANCHOR it by loop closure on cell revisits
(head-direction-cell + landmark correction). Test if this makes absolute heading SIZE-INVARIANT.

Tracker (grid-anchored, sign-corrected): h_pred = h_est + prev_turn; k = round((h_pred-gh)/HALF); h_est = gh+90k.
Re-anchor on revisit to the MOST-RECENT prior visit of the same cell (small time gap -> small relative drift):
  rel = relative rotation since that visit (ORACLE = true; or MATCHER = 95% quadrant + grid fine)
  implied = h_stored + rel ; k = round((implied - gh)/HALF) ; h_est = gh + 90k   (integer correction only)

Compare abs heading err (deg) vs size for: no-reanchor (drifts) | oracle-reanchor | matcher-reanchor.
FLAT & low across sizes => absolute heading is size-invariant WITH loop closure. That flips the verdict.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
from heading_attack.b03d_integrated import train_matcher, matcher_relquad
HALF = math.pi / 2


def run_size(n, n_mazes, seed0, mode, matcher=None):
    """mode in {'none','oracle','matcher'}. Returns abs heading err (deg) and k-wrong fraction."""
    e_h, k_wrong = [], []
    for mi in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + mi, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0; T = 40 * n
        h_est = None; gh0 = None; prev_turn = 0.0
        store = {}   # cell -> (h_est, true_ang, gh, g, l)  most-recent visit
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
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                gh = -_grid_heading(g)                        # sign-corrected (co-rotates with command)
                if h_est is None:
                    h_est = gh; gh0 = gh
                else:
                    h_pred = h_est + prev_turn
                    k = round((h_pred - gh) / HALF); h_est = gh + HALF * k
                cell = (int(env.px), int(env.py))
                # RE-ANCHOR the quadrant on revisit
                if mode != 'none' and cell in store:
                    h_s, ang_s, gh_s, g_s, l_s = store[cell]
                    if mode == 'oracle':
                        rel = wrap(env.ang - ang_s)                       # true relative rotation (mod 2pi)
                    else:  # matcher: grid fine + 95% quadrant
                        rq = matcher_relquad(matcher, g, l, g_s, l_s)     # rel quadrant now vs stored
                        rel = wrap(wrap(gh - gh_s) + rq * HALF)           # fine (grid) + quadrant (matcher)
                    implied = h_s + rel
                    k = round((implied - gh) / HALF); h_est = gh + HALF * k
                if cell not in store:                          # anchor = FIRST visit (best-anchored reference)
                    store[cell] = (h_est, env.ang, gh, g, l)
                true = wrap(env.ang - ang0); est_rel = wrap(h_est - gh0)
                e_h.append(abs(wrap(est_rel - true)))
                ktrue = round((true - wrap(gh - gh0)) / HALF); kest = round((est_rel - wrap(gh - gh0)) / HALF)
                k_wrong.append(1.0 if (ktrue % 4) != (kest % 4) else 0.0)
                prev_turn = turn; motion(env, a); steps += 1
    return np.mean(e_h) * 180 / math.pi, float(np.mean(k_wrong))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64, 89, 129])
    ap.add_argument("--mazes", type=int, default=3)
    a = ap.parse_args()
    print("Training matcher (n=6,12)..."); m = train_matcher(epochs=1200)
    print("b07: heading with quadrant RE-ANCHORING via loop closure. abs err (deg) [k-wrong%] vs size.")
    print(f"{'n':>4} {'grid':>9} | {'no-reanchor':>14} {'ORACLE-reanch':>16} {'MATCHER-reanch':>16}")
    for n in a.sizes:
        en, kn = run_size(n, a.mazes, 5000, 'none')
        eo, ko = run_size(n, a.mazes, 5000, 'oracle')
        em, km = run_size(n, a.mazes, 5000, 'matcher', m)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {en:7.1f} [{kn*100:4.0f}%] {eo:8.1f} [{ko*100:4.0f}%] {em:8.1f} [{km*100:4.0f}%]")
    print("\nORACLE/MATCHER flat & low across size => absolute heading SIZE-INVARIANT with loop closure (HD-cell story).")
