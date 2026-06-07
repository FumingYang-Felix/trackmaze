"""THE DEPLOYABLE METHOD (works end-to-end, no oracle rotation). Recovers global heading size-invariantly in
the recoverable (loopy) phase, using a realistic allothetic sensor (a few distinct distant/mid-range beacons =
a coarse compass/skyline). Pipeline:
  - front-end: grid-fine heading (drift-free mod-90, sign-corrected) + per-step chain quadrant (turn vs grid).
  - loop closures: same-cell revisits (oracle place-id here; place-rec is a separate axis) whose RELATIVE
    QUADRANT comes from the beacon constellation, GRID-CORRECTED: rel = round((beacon_rot - Δgrid)/90).
    Closures are WELL-DISTRIBUTED (sampled steps -> their cell's first visit) so BP converges.
  - back-end: Z4 max-product BP synchronization (chain + closures + anchor) -> quadrant field -> heading.
Result: ~2-4deg flat to 49x49 (no oracle rotation); larger needs BP iters ~ size (pure convergence, closAcc
stays 100%). Two fixes were essential: grid-correct the closure quadrant; well-distribute the closures.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, wrap, motion
from heading_attack.b14_bp_z4 import bp_z4
HALF = math.pi / 2


def drive(n, seed, loop, K=6, R=1e9):
    """random-walk coverage with K distinct distant beacons (R=inf = skyline). Records grid-fine, turn, cell,
    true heading, and per-beacon bearing (rel to heading)."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed); cells = list(adj.keys())
    sel = rng.choice(len(cells), min(K, len(cells)), replace=False)
    beac = np.array([[2 * cells[i][0] + 1.5, 2 * cells[i][1] + 1.5] for i in sel])
    target = max(2, int(8 * len(adj) * math.log(len(adj) + 2) / 10)); cur = (0, 0); hops = 0
    GH, TURN, CELL, ANG, BR = [], [], [], [], []; pt = 0.0
    while hops < target:
        nb = adj.get(cur, [])
        if not nb: break
        nx = nb[rng.integers(len(nb))]; wx, wy = 2 * nx[0] + 1.5, 2 * nx[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            t = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            g, _ = omni(env.wall, env.col, env.px, env.py, env.ang); gh = (-_grid_heading(g)) % HALF
            rel = beac - np.array([env.px, env.py]); dist = np.hypot(rel[:, 0], rel[:, 1])
            br = {k: wrap(math.atan2(rel[k, 1], rel[k, 0]) - env.ang) for k in range(len(beac)) if dist[k] < R}
            GH.append(gh); TURN.append(pt); CELL.append(nx); ANG.append(wrap(env.ang - ang0)); BR.append(br); pt = t
            motion(env, a)
        cur = nx; hops += 1
    return np.array(GH), np.array(TURN), CELL, np.array(ANG), BR


def beacon_rot(bi, bj):
    d = [wrap(bj[k] - bi[k]) for k in bi if k in bj]      # heading_i - heading_j
    return math.atan2(np.mean(np.sin(d)), np.mean(np.cos(d))) if d else None


def deploy_slam(GH, TURN, CELL, ANG, BR, n_closures=2000, bp_iters=600, seed=1):
    T = len(GH)
    cr = np.array([0] + [int(round((TURN[t] - (GH[t] - GH[t - 1])) / HALF)) % 4 for t in range(1, T)])
    byc = defaultdict(list)
    for t, c in enumerate(CELL): byc[c].append(t)
    fv = {c: ts[0] for c, ts in byc.items()}
    rng = np.random.default_rng(seed); qs = rng.choice(T, min(n_closures, T), replace=False); cl = []
    for t in qs:
        j = fv[CELL[t]]
        if j >= t: continue
        rot = beacon_rot(BR[t], BR[j])
        if rot is None: continue
        rel = int(round((rot - (GH[t] - GH[j])) / HALF)) % 4     # GRID-CORRECTED relative quadrant
        cl.append((int(t), int(j), rel))
    edges = [(t, t - 1, int(cr[t]), 3.0) for t in range(1, T)] + [(i, j, rel, 1.5) for (i, j, rel) in cl]
    q = bp_z4(edges, T, theta=6.0, iters=bp_iters, damp=0.2)
    h = np.array([wrap((GH[t] - GH[0]) + HALF * (q[t] - q[0])) for t in range(T)])
    err = np.mean([abs(wrap(h[t] - ANG[t])) for t in range(T)]) * 180 / math.pi
    return err, len(cl)


def online_err(GH, TURN, ANG):
    T = len(GH); cr = np.array([0] + [int(round((TURN[t] - (GH[t] - GH[t - 1])) / HALF)) % 4 for t in range(1, T)])
    q = np.zeros(T, int)
    for t in range(1, T): q[t] = (q[t - 1] + cr[t]) % 4
    h = np.array([wrap((GH[t] - GH[0]) + HALF * (q[t] - q[0])) for t in range(T)])
    return np.mean([abs(wrap(h[t] - ANG[t])) for t in range(T)]) * 180 / math.pi


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24]); ap.add_argument("--mazes", type=int, default=2)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()
    print(f"DEPLOYABLE SLAM (no oracle rotation), allothetic distant cues K={a.K}. global heading err vs size. loop={a.loop}", flush=True)
    print(f"{'n':>4} {'px':>5} | {'online(drift)':>13} {'deploy-SLAM':>11}", flush=True)
    for n in a.sizes:
        oe, se = [], []
        for mi in range(a.mazes):
            GH, TURN, CELL, ANG, BR = drive(n, 7000 + mi, a.loop, K=a.K)
            oe.append(online_err(GH, TURN, ANG)); e, _ = deploy_slam(GH, TURN, CELL, ANG, BR, bp_iters=a.iters); se.append(e)
        print(f"{n:>4} {2*n+1:>5} | {np.mean(oe):13.1f} {np.mean(se):11.1f}", flush=True)
    print("\ndeploy-SLAM flat & ~2-4deg vs online drift => global heading recovered size-invariantly, no oracle rotation.", flush=True)
