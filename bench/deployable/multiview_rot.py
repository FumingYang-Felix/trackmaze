"""Attack the matcher wall (55-65% on loopy) with a MULTI-VIEW rotation estimator. Key asset: the grid-fine
heading (drift-free mod-90) is known every step -> de-rotate each omni view into the WALL FRAME, which removes
the continuous rotation and leaves ONLY the 4-way quadrant. Averaging the grid-canonicalized views over a short
WINDOW around each visit denoises the (weak, open-maze) per-view signal. Relative quadrant between two visits =
argmax over r in {0,1,2,3} of correlation(meanCanon_A, roll(meanCanon_B, r*KO/4)).

Compares single-view cross-corr (the old matcher's cue) vs multi-view grid-canon, quad accuracy vs size, loopy.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, wrap, motion
HALF = math.pi / 2
Q = KO // 4   # rays per quadrant (=8)


def drive(n, seed, loop, lmd=0.15):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=lmd, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed)
    target = max(2, int(8 * len(adj) * math.log(len(adj) + 2) / 10))
    G, L, GH, CELL, ANG = [], [], [], [], []; cur = (0, 0); hops = 0
    while hops < target:
        nb = adj.get(cur, [])
        if not nb: break
        nx = nb[rng.integers(len(nb))]; wx, wy = 2 * nx[0] + 1.5, 2 * nx[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            G.append(g); L.append(l); GH.append((-_grid_heading(g)) % HALF); CELL.append(nx); ANG.append(wrap(env.ang - ang0))
            motion(env, a)
        cur = nx; hops += 1
    return np.array(G), np.array(L), np.array(GH), CELL, np.array(ANG)


def canon(g, gh):
    """roll omni view into the wall frame using grid-fine heading (removes the continuous mod-90 rotation)."""
    sh = int(round(gh / (2 * math.pi) * KO))
    return np.roll(g, sh)


def precompute_canon(G, L, GH):
    GC = np.array([canon(G[t], GH[t]) for t in range(len(G))])
    LC = np.array([canon(L[t], GH[t]) for t in range(len(G))])
    return GC, LC


def mean_canon(GC, LC, t, W):
    a, b = max(0, t - W), min(len(GC), t + W + 1)
    return GC[a:b].mean(0), LC[a:b].mean(0)


def relquad_single(g1, l1, g2, l2):
    c = np.array([np.dot(g1, np.roll(g2, s)) + np.dot(l1, np.roll(l2, s)) for s in range(KO)])
    s = int(np.argmax(c)); return int(round((s if s <= KO // 2 else s - KO) / Q)) % 4


def relquad_multi(gc1, lc1, gc2, lc2):
    """4-way: the views are already grid-canon (continuous part removed), so only test the 4 quadrant rolls."""
    sc = [np.dot(gc1, np.roll(gc2, r * Q)) + np.dot(lc1, np.roll(lc2, r * Q)) for r in range(4)]
    return int(np.argmax(sc))


def eval_size(n, loop, seed0, n_mazes=3, W=6, lmd=0.15):
    sing, mult = [], []
    for mi in range(n_mazes):
        G, L, GH, CELL, ANG = drive(n, seed0 + mi, loop, lmd)
        GC, LC = precompute_canon(G, L, GH)
        bycell = defaultdict(list)
        for t, c in enumerate(CELL): bycell[c].append(t)
        revis = [c for c, v in bycell.items() if len(v) >= 2]
        rng = np.random.default_rng(0); ns = ms = tot = 0
        for _ in range(2000):
            if not revis: break
            ts = bycell[revis[rng.integers(len(revis))]]; i, j = rng.choice(len(ts), 2, replace=False)
            ti, tj = ts[i], ts[j]
            true = int(round(wrap(ANG[ti] - ANG[tj]) / HALF)) % 4
            if relquad_single(G[ti], L[ti], G[tj], L[tj]) == true: ns += 1
            g1, l1 = mean_canon(GC, LC, ti, W); g2, l2 = mean_canon(GC, LC, tj, W)
            # relative quadrant of canon frames; canon already aligned the fine part, so add back the quad of raw
            if relquad_multi(g1, l1, g2, l2) == true: ms += 1
            tot += 1
        sing.append(ns / max(tot, 1)); mult.append(ms / max(tot, 1))
    return np.mean(sing), np.mean(mult)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32]); ap.add_argument("--W", type=int, default=6)
    a = ap.parse_args()
    print(f"Multi-view (grid-canon, window-avg) rotation estimator vs single-view. loop={a.loop}, W={a.W}.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'single':>7} {'multi-view':>11}", flush=True)
    for n in a.sizes:
        s, m = eval_size(n, a.loop, 9000, W=a.W)
        print(f"{n:>4} {2*n+1:>5} | {s*100:6.0f}% {m*100:10.0f}%", flush=True)
    print("\nmulti-view >> single & toward 90% => the matcher wall is the per-view weakness; window+grid-canon fixes it.", flush=True)
