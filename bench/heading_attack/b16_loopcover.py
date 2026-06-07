"""b16: concrete solver-level confirmation of the phase transition + the COVERAGE nuance. The b15 floor uses
the full cell-adjacency graph -> it assumes the trajectory TRAVERSES the loops. DFS only walks a spanning tree
(stays 1D even in a loopy maze). So recoverability needs BOTH (a) a loopy environment AND (b) a loop-CLOSING
trajectory. Here: a RANDOM-WALK coverage policy (naturally traverses loop edges -> 2D closure graph), oracle
Z4 closures, loopy BP (converges fast on a 2D graph, unlike the 1D chain). Measure q-acc & heading err vs size
for tree(loop=0) vs loopy(loop=0.9). Loopy+coverage FLAT high q-acc => recoverable size-invariantly (estimator
confirms the floor). Tree decays => unrecoverable. This is the active-SLAM explore-vs-localize tradeoff made exact.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, wrap, motion
from heading_attack.b14_bp_z4 import bp_z4
HALF = math.pi / 2


def rw_drive(n, seed, loop, cover_mult=8):
    """Random-walk coverage: at each cell pick a random open neighbor (revisits + traverses loop edges)."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); ncell = len(adj)
    rng = np.random.default_rng(seed)
    target_cells = max(2, int(cover_mult * ncell * math.log(ncell + 2) / 10))  # ~coverage budget (cell hops)
    rec = []; prev_turn = 0.0; cur = (0, 0); hops = 0
    while hops < target_cells:
        nbrs = adj.get(cur, [])
        if not nbrs: break
        nxt = nbrs[rng.integers(len(nbrs))]
        wx, wy = 2 * nxt[0] + 1.5, 2 * nxt[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            gh = (-_grid_heading(g)) % HALF
            rec.append((gh, prev_turn, nxt, wrap(env.ang - ang0)))   # label by target cell (cell coords)
            prev_turn = turn; motion(env, a)
        cur = nxt; hops += 1
    return rec


def oracle_edges(rec, npc=10):
    T = len(rec); gh = np.array([r[0] for r in rec])
    qtrue = np.array([int(round((rec[t][3] - rec[t][0] + rec[0][0]) / HALF)) for t in range(T)])
    edges = []
    for t in range(1, T):
        edges.append((t, t - 1, int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF)) % 4, 3.0))
    bycell = defaultdict(list)
    for t, rr in enumerate(rec): bycell[rr[2]].append(t)
    for c, ts in bycell.items():
        for a in ts[:4]:
            for b in ts[1:1 + npc]:
                if b <= a: continue
                edges.append((b, a, (qtrue[b] - qtrue[a]) % 4, 1.0))
    return edges, qtrue, gh


def ev(n, loop, mazes=2):
    accs, errs, Ts = [], [], []
    for mi in range(mazes):
        rec = rw_drive(n, 5000 + mi, loop); T = len(rec)
        if T < 5: continue
        edges, qtrue, gh = oracle_edges(rec)
        q = bp_z4(edges, T, theta=8.0, iters=200, damp=0.2)
        acc = max(np.mean((q - qtrue - o) % 4 == 0) for o in range(4))
        qa = (q - q[0]) % 4
        est = np.array([wrap((gh[t] - gh[0]) + HALF * qa[t]) for t in range(T)])
        tru = np.array([r[3] for r in rec])
        accs.append(acc * 100); errs.append(np.mean([abs(wrap(est[t] - tru[t])) for t in range(T)]) * 180 / math.pi); Ts.append(T)
    return np.mean(accs), np.mean(errs), int(np.mean(Ts))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--mazes", type=int, default=2); a = ap.parse_args()
    print("b16: random-walk COVERAGE + oracle Z4 closures + loopy BP. q-acc & head-err vs size, tree vs loopy.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'TREE(loop0) acc/err':>22} | {'LOOPY(loop.9) acc/err':>22}", flush=True)
    for n in a.sizes:
        at, et, Tt = ev(n, 0.0, a.mazes); al, el, Tl = ev(n, 0.9, a.mazes)
        print(f"{n:>4} {2*n+1:>5} | {at:8.1f}% {et:7.1f}deg (T{Tt}) | {al:8.1f}% {el:7.1f}deg (T{Tl})", flush=True)
    print("\nLOOPY high acc & low err, FLAT with size => global frame recoverable size-invariantly (solver confirms", flush=True)
    print("b15 floor). TREE acc decays / err grows => unrecoverable. Coverage (loop-closing walk) is required.", flush=True)
