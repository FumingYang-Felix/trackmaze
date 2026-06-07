"""b03b: full heading estimator combining the three signals, to resolve the quadrant.
  - PRIOR: command integration (each step += commanded turn; drift-free relative to commands)
  - FINE: grid mod-90 snap (freq-4; drift-free, ~20deg noise) -> pins the continuous part
  - QUADRANT re-anchor: on revisiting a cell (oracle place-id for this mechanism test), circular cross-correlate
    the current omni view with the stored one -> relative true rotation -> implied current heading =
    stored_heading + rel; snap that to the grid-consistent value; pull h_est toward it. This re-anchors the
    drifting quadrant to an earlier (more accurate) visit. Robust variant: only accept the closure if it agrees
    with the grid (the relative rotation must be ~a multiple of 90 after removing the grid-fine part).

Measures absolute heading error (deg, full circle) vs size, vs the round16 learned estimator (67deg@n64) and
the cmd / grid-only baselines. If b03b << that and flat with size -> the quadrant is resolved -> heading solved.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion

HALF = math.pi / 2


def run_size(n, n_mazes, seed0, gain=0.5, store_thr=None):
    errs_full, errs_grid, errs_cmd = [], [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang; cmd = 0.0; h_est = 0.0
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
        store = {}  # cell -> (h_est_at_store, geo)
        steps = 0; T = 40 * n
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                err = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(err) > 0.20 and err > 0) else (2 if abs(err) > 0.20 else 0)
                g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
                gh = _grid_heading(g)                                  # true heading mod 90 (drift-free)
                # FINE: snap h_est's mod-90 part to the grid observation
                k = round((h_est - gh) / HALF); h_est = h_est + 0.6 * wrap(gh + k * HALF - h_est)
                cell = (int(env.px), int(env.py))
                # QUADRANT re-anchor via cross-corr loop closure (oracle place-id)
                if cell in store:
                    h_s, g_s = store[cell]
                    c = np.array([np.dot(g, np.roll(g_s, s)) for s in range(KO)])  # rel rotation g_s->g
                    s = int(np.argmax(c)); rel = 2 * math.pi * (s - KO if s > KO // 2 else s) / KO
                    implied = h_s + rel
                    kk = round((implied - gh) / HALF); implied = gh + kk * HALF    # snap to grid-consistent
                    h_est = h_est + gain * wrap(implied - h_est)
                store[cell] = (h_est, g)
                # record errors
                tru = wrap(env.ang - ang0)
                errs_full.append(abs(wrap(h_est - tru)))
                errs_grid.append(min(abs(wrap(gh - tru % HALF)), HALF))   # grid alone (mod-90, used as absolute=ambiguous)
                errs_cmd.append(abs(wrap(cmd - tru)))
                if a == 2: cmd -= 0.20; h_est -= 0.20
                elif a == 3: cmd += 0.20; h_est += 0.20
                motion(env, a); steps += 1
    D = lambda e: np.mean(e) * 180 / math.pi
    return D(errs_full), D(errs_cmd)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--mazes", type=int, default=4); ap.add_argument("--gain", type=float, default=0.5)
    a = ap.parse_args()
    print(f"b03b full heading estimator (cmd prior + grid mod-90 + cross-corr loop-closure quadrant, oracle place-id)")
    print(f"{'n':>4} {'grid':>8} | {'b03b full':>9} {'cmd only':>9}  (deg; vs round16 learned 30->67)")
    for n in a.sizes:
        full, cmd = run_size(n, a.mazes, 5000, a.gain)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {full:9.1f} {cmd:9.1f}")
    print("\nb03b << 67 and flat with size => quadrant resolved via loop closure => heading SOLVED (with place-id).")
