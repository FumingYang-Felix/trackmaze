"""Richer sensor (the fallback): the env's raycast carries binned landmark TYPES (LBINS<=12 for ambiguity=1),
but the current omni collapses them to binary presence -- throwing away exactly the rotation-breaking info.
Here the omni returns the landmark TYPE per ray (one-hot). The arrangement of landmark types around a cell is
rotationally DISTINCTIVE, so a cross-correlation reads the relative rotation sharply -- and it transfers OOD
because it matches the PATTERN, not absolute positions. Test: does the rich sensor break the rotation aliasing
(quadrant accuracy >> the 40% binary ceiling, flat with size)?"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import KO, _grid_heading
from round3a import cell_graph, wrap, motion
HALF = math.pi / 2; Q = KO // 4


def cast_rich(wall, col, px, py, theta, maxd=24.0):
    dx, dy = math.cos(theta), math.sin(theta); W = wall.shape[0]
    mX, mY = int(px), int(py)
    ddx = abs(1 / dx) if dx else 1e9; ddy = abs(1 / dy) if dy else 1e9
    if dx < 0: sx = -1; sdx = (px - mX) * ddx
    else:      sx = 1;  sdx = (mX + 1 - px) * ddx
    if dy < 0: sy = -1; sdy = (py - mY) * ddy
    else:      sy = 1;  sdy = (mY + 1 - py) * ddy
    side = 0
    for _ in range(300):
        if sdx < sdy: sdx += ddx; mX += sx; side = 0
        else:         sdy += ddy; mY += sy; side = 1
        if not (0 <= mY < W and 0 <= mX < W): return 1.0, -1
        if wall[mY, mX] > 0:
            pd = (sdx - ddx) if side == 0 else (sdy - ddy)
            return min(pd, maxd) / maxd, int(col[mY, mX] - 1)   # landmark type id (-1 plain wall)
    return 1.0, -1


def omni_rich(wall, col, px, py, ang, LBINS):
    g = np.empty(KO, np.float32); lid = np.empty(KO, np.int32)
    for i in range(KO):
        g[i], lid[i] = cast_rich(wall, col, px, py, ang + 2 * math.pi * i / KO)
    onehot = np.zeros((KO, LBINS), np.float32)
    for i in range(KO):
        if lid[i] >= 0: onehot[i, lid[i] % LBINS] = 1.0
    return g, onehot


def drive(n, seed, loop, lmd=0.15):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=lmd, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; LB = env.LBINS; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed)
    target = max(2, int(8 * len(adj) * math.log(len(adj) + 2) / 10))
    G, LH, BIN, CELL, ANG = [], [], [], [], []; cur = (0, 0); hops = 0
    while hops < target:
        nb = adj.get(cur, [])
        if not nb: break
        nx = nb[rng.integers(len(nb))]; wx, wy = 2 * nx[0] + 1.5, 2 * nx[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            g, oh = omni_rich(env.wall, env.col, env.px, env.py, env.ang, LB)
            G.append(g); LH.append(oh); BIN.append((-_grid_heading(g)) % HALF); CELL.append(nx); ANG.append(wrap(env.ang - ang0))
            motion(env, a)
        cur = nx; hops += 1
    return np.array(G), np.array(LH), np.array(BIN), CELL, np.array(ANG), LB


def relquad_bin(g1, b1, g2, b2):
    """OLD binary sensor: cross-corr on dist + binary landmark presence."""
    bp1 = (b1.sum(1) > 0).astype(np.float32); bp2 = (b2.sum(1) > 0).astype(np.float32)
    c = np.array([np.dot(g1, np.roll(g2, s)) + np.dot(bp1, np.roll(bp2, s)) for s in range(KO)])
    s = int(np.argmax(c)); return int(round((s if s <= KO // 2 else s - KO) / Q)) % 4


def relquad_rich(g1, oh1, g2, oh2):
    """RICH sensor: cross-corr on dist + per-TYPE one-hot landmark channel (rotationally distinctive)."""
    best, bs = 0, -1e18
    for s in range(KO):
        sc = np.dot(g1, np.roll(g2, s)) + float((oh1 * np.roll(oh2, s, axis=0)).sum())
        if sc > bs: bs = sc; best = s
    return int(round((best if best <= KO // 2 else best - KO) / Q)) % 4


def eval_size(n, loop, seed0, n_mazes=3, lmd=0.15):
    rb, rr = [], []
    for mi in range(n_mazes):
        G, LH, BIN, CELL, ANG, LB = drive(n, seed0 + mi, loop, lmd)
        bycell = defaultdict(list)
        for t, c in enumerate(CELL): bycell[c].append(t)
        rev = [c for c, v in bycell.items() if len(v) >= 2]; rng = np.random.default_rng(0)
        ob = orr = tot = 0
        for _ in range(2000):
            if not rev: break
            ts = bycell[rev[rng.integers(len(rev))]]; i, j = rng.choice(len(ts), 2, replace=False); ti, tj = ts[i], ts[j]
            true = int(round(wrap(ANG[ti] - ANG[tj]) / HALF)) % 4
            if relquad_bin(G[ti], LH[ti], G[tj], LH[tj]) == true: ob += 1
            if relquad_rich(G[ti], LH[ti], G[tj], LH[tj]) == true: orr += 1
            tot += 1
        rb.append(ob / max(tot, 1)); rr.append(orr / max(tot, 1))
    return np.mean(rb), np.mean(rr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32]); ap.add_argument("--lmd", type=float, default=0.15)
    a = ap.parse_args()
    print(f"RICH sensor (landmark TYPE per ray) vs binary, rotation-quadrant accuracy. loop={a.loop} lmd={a.lmd}.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'binary':>7} {'rich':>7}", flush=True)
    for n in a.sizes:
        b, r = eval_size(n, a.loop, 9000, lmd=a.lmd)
        print(f"{n:>4} {2*n+1:>5} | {b*100:6.0f}% {r*100:6.0f}%", flush=True)
    print("\nrich >> binary & toward 90% flat => richer sensor breaks rotation aliasing => deployable SLAM unlocks.", flush=True)
