"""b03 KERNEL test: can a circular cross-correlation of two omni views of the SAME cell recover their
RELATIVE true heading? If yes, loop closure can propagate absolute heading (resolve the quadrant) -- the
core mechanism of branch b03.

Theory: at a fixed place, the omni view at true heading theta = the surroundings circularly shifted by theta.
Two visits at theta1, theta2 -> views shifted by (theta2-theta1). Circular cross-correlation argmax recovers
that shift. Combined with the drift-free grid (mod 90), the shift's continuous part is known, so we'd only
need the discrete 90-multiple. Here we test the RAW recovery: does argmax cross-corr ~= true (theta2-theta1)?

Reports: median/mean absolute error of the recovered relative heading (deg) for same-cell visit pairs, vs
size. Also a DIFFERENT-cell control (should be near-random) to confirm it's place-specific. <~ half a ray
bin (5.6deg) = excellent; if errors cluster at multiples of 90 it's the grid symmetry (expected, resolvable).
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, dfs_walk, wrap, motion


def collect(n, n_mazes, seed0, T):
    """Drive DFS traversal; record per cell-visit: cell, true heading-from-start, RAW omni geo (KO,)."""
    visits = []  # (maze, cell, theta_rel, geo)
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                a = 3 if (abs(wrap(math.atan2(dy, dx) - env.ang)) > 0.20 and wrap(math.atan2(dy, dx) - env.ang) > 0) \
                    else (2 if abs(wrap(math.atan2(dy, dx) - env.ang)) > 0.20 else 0)
                g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
                visits.append((m, (int(env.px), int(env.py)), wrap(env.ang - ang0), g))
                motion(env, a); steps += 1
    return visits


def xcorr_shift(g1, g2):
    """Circular cross-correlation argmax -> estimated rotation (rad) mapping g1->g2 (g2 ~ roll(g1, +shift))."""
    c = np.array([np.dot(g2, np.roll(g1, s)) for s in range(KO)])
    s = int(np.argmax(c))
    if s > KO // 2: s -= KO
    return 2 * math.pi * s / KO


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44])
    ap.add_argument("--mazes", type=int, default=4); ap.add_argument("--pairs", type=int, default=2000)
    a = ap.parse_args()
    print("b03 kernel: cross-corr recovered relative heading error (deg) for SAME-cell pairs vs DIFFERENT-cell")
    print(f"{'n':>4} {'grid':>8} | {'same med':>9} {'same mean':>9} {'%<11deg':>8} {'%near90k':>9} | {'diff med':>9}")
    for n in a.sizes:
        vis = collect(n, a.mazes, 5000, 40 * n)
        bycell = defaultdict(list)
        for v in vis: bycell[(v[0], v[1])].append(v)
        rng = np.random.default_rng(0)
        same_err, diff_err = [], []
        revis = [c for c, vs in bycell.items() if len(vs) >= 2]
        tries = 0
        while len(same_err) < a.pairs and tries < a.pairs * 30 and revis:
            tries += 1; c = revis[rng.integers(len(revis))]; vs = bycell[c]
            i, j = rng.choice(len(vs), 2, replace=False)
            true_rel = wrap(vs[j][2] - vs[i][2]); est = xcorr_shift(vs[i][3], vs[j][3])
            same_err.append(abs(wrap(est - true_rel)) * 180 / math.pi)
        allv = vis
        while len(diff_err) < len(same_err):
            v1, v2 = allv[rng.integers(len(allv))], allv[rng.integers(len(allv))]
            if v1[1] != v2[1] or v1[0] != v2[0]:
                true_rel = wrap(v2[2] - v1[2]); est = xcorr_shift(v1[3], v2[3])
                diff_err.append(abs(wrap(est - true_rel)) * 180 / math.pi)
        se = np.array(same_err); de = np.array(diff_err)
        near90 = np.mean(np.minimum(se % 90, 90 - (se % 90)) < 11) * 100   # fraction whose error is a ~multiple of 90
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {np.median(se):9.1f} {np.mean(se):9.1f} {np.mean(se<11)*100:7.0f}% {near90:8.0f}% | {np.median(de):9.1f}")
    print("\nsame-cell error small (or clustered at multiples of 90 -> grid-resolvable) => cross-corr recovers")
    print("relative heading => loop closure can propagate the quadrant. diff-cell ~random => it's place-specific.")
