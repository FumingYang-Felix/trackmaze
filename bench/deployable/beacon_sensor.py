"""The genuine richer sensor: FREE-STANDING BEACONS in open space (objects/pillars), visible across open cells
by line-of-sight, each a distinct type. Wall-bound landmarks vanish in loopy/open mazes (no walls); beacons
provide rotation cues exactly where walls don't -- resolving the recoverability(open) vs rotation-cue(structure)
tension. The local beacon constellation (bearings x types) is rotationally distinctive; cross-correlation reads
the relative rotation, transfers OOD (pattern, not absolute). Beacons are LOCAL (range R, LOS-blocked) so they
are NOT a global compass (they don't collapse the global 2-bit symmetry; that stays per the phase diagram).

Test: rotation-quadrant accuracy with beacons vs the binary wall sensor, vs size, in loopy mazes.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, wrap, motion
HALF = math.pi / 2; Q = KO // 4


def place_beacons(adj, rng, density, ntypes):
    cells = list(adj.keys())
    k = max(2, int(density * len(cells)))
    sel = rng.choice(len(cells), min(k, len(cells)), replace=False)
    return [(cells[i], int(rng.integers(ntypes))) for i in sel]   # ((cx,cy), type)


def los_clear(wall, x0, y0, x1, y1):
    """coarse line-of-sight: sample along segment, blocked if a wall cell is hit."""
    d = math.hypot(x1 - x0, y1 - y0); steps = int(d * 3) + 1
    for s in range(1, steps):
        u = s / steps; x = x0 + (x1 - x0) * u; y = y0 + (y1 - y0) * u
        if wall[int(y), int(x)] > 0: return False
    return True


def beacon_omni(wall, beacons, px, py, ang, ntypes, R=6.0):
    """one-hot (KO x ntypes) of nearest in-range LOS beacon per bearing bin."""
    oh = np.zeros((KO, ntypes), np.float32); best = np.full(KO, 1e9)
    for (cx, cy), tp in beacons:
        bx, by = 2 * cx + 1.5, 2 * cy + 1.5; dx, dy = bx - px, by - py
        dist = math.hypot(dx, dy)
        if dist > R or dist < 0.2: continue
        if not los_clear(wall, px, py, bx, by): continue
        bear = (math.atan2(dy, dx) - ang) % (2 * math.pi); b = int(round(bear / (2 * math.pi) * KO)) % KO
        if dist < best[b]: best[b] = dist; oh[b] = 0.0; oh[b, tp] = 1.0
    return oh


def drive(n, seed, loop, bdens=0.25, ntypes=8, R=6.0):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed)
    beacons = place_beacons(adj, rng, bdens, ntypes)
    target = max(2, int(8 * len(adj) * math.log(len(adj) + 2) / 10))
    G, BC, BIN, CELL, ANG = [], [], [], [], []; cur = (0, 0); hops = 0
    while hops < target:
        nb = adj.get(cur, [])
        if not nb: break
        nx = nb[rng.integers(len(nb))]; wx, wy = 2 * nx[0] + 1.5, 2 * nx[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
            bo = beacon_omni(env.wall, beacons, env.px, env.py, env.ang, ntypes, R)
            G.append(g); BC.append(bo); BIN.append((-_grid_heading(g)) % HALF); CELL.append(nx); ANG.append(wrap(env.ang - ang0))
            motion(env, a)
        cur = nx; hops += 1
    return np.array(G), np.array(BC), np.array(BIN), CELL, np.array(ANG)


def relquad_wall(g1, g2):
    c = np.array([np.dot(g1, np.roll(g2, s)) for s in range(KO)])
    s = int(np.argmax(c)); return int(round((s if s <= KO // 2 else s - KO) / Q)) % 4


def relquad_beacon(g1, bc1, g2, bc2):
    best, bs = 0, -1e18
    for s in range(KO):
        sc = 0.3 * np.dot(g1, np.roll(g2, s)) + float((bc1 * np.roll(bc2, s, axis=0)).sum())
        if sc > bs: bs = sc; best = s
    return int(round((best if best <= KO // 2 else best - KO) / Q)) % 4


def eval_size(n, loop, seed0, n_mazes=3, bdens=0.25, ntypes=8, R=6.0):
    rw, rb, cov = [], [], []
    for mi in range(n_mazes):
        G, BC, BIN, CELL, ANG = drive(n, seed0 + mi, loop, bdens, ntypes, R)
        cov.append(float(np.mean([bc.sum() > 0 for bc in BC])))   # frac of steps that see >=1 beacon
        bycell = defaultdict(list)
        for t, c in enumerate(CELL): bycell[c].append(t)
        rev = [c for c, v in bycell.items() if len(v) >= 2]; rng = np.random.default_rng(0)
        ow = ob = tot = 0
        for _ in range(2000):
            if not rev: break
            ts = bycell[rev[rng.integers(len(rev))]]; i, j = rng.choice(len(ts), 2, replace=False); ti, tj = ts[i], ts[j]
            true = int(round(wrap(ANG[ti] - ANG[tj]) / HALF)) % 4
            if relquad_wall(G[ti], G[tj]) == true: ow += 1
            if relquad_beacon(G[ti], BC[ti], G[tj], BC[tj]) == true: ob += 1
            tot += 1
        rw.append(ow / max(tot, 1)); rb.append(ob / max(tot, 1))
    return np.mean(rw), np.mean(rb), np.mean(cov)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--bdens", type=float, default=0.25); ap.add_argument("--ntypes", type=int, default=8); ap.add_argument("--R", type=float, default=6.0)
    a = ap.parse_args()
    print(f"BEACON sensor (free-standing, range R={a.R}) vs wall-only, rotation-quadrant acc. loop={a.loop} bdens={a.bdens} ntypes={a.ntypes}.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'wall-only':>9} {'beacon':>7} {'beaconCov':>10}", flush=True)
    for n in a.sizes:
        w, b, c = eval_size(n, a.loop, 9000, bdens=a.bdens, ntypes=a.ntypes, R=a.R)
        print(f"{n:>4} {2*n+1:>5} | {w*100:8.0f}% {b*100:6.0f}% {c*100:9.0f}%", flush=True)
    print("\nbeacon >> wall & toward 90% flat => free-standing beacons break the rotation aliasing in open mazes.", flush=True)
