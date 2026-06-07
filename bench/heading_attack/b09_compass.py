"""b09: confirm the flip side. b08 proved the absolute quadrant is unobservable from local views (exact 4-fold
symmetry). CLAIM: one OMNIPRESENT orientation cue (a 'compass'/'sun'/direction-colored walls) -- even a NOISY,
ERROR-PRONE one -- collapses the quadrant ambiguity and makes FULL absolute heading size-invariant (since the
fine mod-90 part is already drift-free). Model the cue as an absolute-quadrant observation with error prob p
(p=0 perfect compass; p=0.3 a weak/occasionally-wrong featural cue). Re-anchor the grid-tracked integer to it.

If heading error is FLAT with size for p<1 (and ~grid floor as p->0) => the ONLY thing blocking size-invariant
absolute heading is the missing global cue; the environment-symmetry is the whole story; nothing else is
fundamental. (No compass => degenerates to b06, which drifts.)
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
HALF = math.pi / 2


def run_size(n, n_mazes, seed0, p_err):
    """Grid-anchored fine + per-step compass (abs quadrant, wrong w.p. p_err) -> integer. Abs heading err deg."""
    e_h = []
    for mi in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + mi, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang; rng = np.random.default_rng(777 + mi)
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0; T = 40 * n
        gh0 = None
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                e = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
                g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
                gh = -_grid_heading(g)                          # drift-free fine (sign-corrected)
                if gh0 is None: gh0 = gh
                # COMPASS: true absolute world quadrant, wrong w.p. p_err (the featural cue)
                qtrue = int(math.floor((env.ang % (2 * math.pi)) / HALF)) % 4
                qobs = qtrue if rng.random() > p_err else (qtrue + rng.integers(1, 4)) % 4
                # combine: fine (grid) + quadrant (compass). q0 reference at start removes the global constant.
                if steps == 0: q0 = qobs
                h_abs = gh + HALF * ((qobs - q0))               # heading rel start via wall-fine + compass-quadrant
                true = wrap(env.ang - ang0)
                e_h.append(abs(wrap(wrap(h_abs - gh0) - true)))
                motion(env, a); steps += 1
    return np.mean(e_h) * 180 / math.pi


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64, 89, 129])
    ap.add_argument("--mazes", type=int, default=3)
    a = ap.parse_args()
    print("b09: full absolute heading = drift-free grid mod-90 + an omnipresent (noisy) compass quadrant. err(deg).")
    print(f"{'n':>4} {'grid':>9} | {'p=0 (perfect)':>14} {'p=0.1':>8} {'p=0.3':>8} {'no-compass(b06)':>16}")
    for n in a.sizes:
        e0 = run_size(n, a.mazes, 5000, 0.0); e1 = run_size(n, a.mazes, 5000, 0.1); e3 = run_size(n, a.mazes, 5000, 0.3)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {e0:14.1f} {e1:8.1f} {e3:8.1f} {'(drifts ~85)':>16}")
    print("\nFLAT with size for p<1 => the missing global orientation cue is the WHOLE story; nothing else is fundamental.")
