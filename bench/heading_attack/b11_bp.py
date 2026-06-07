"""b11: discrete Z4 quadrant recovery done RIGHT. Spectral (b10) failed because the traversal graph is a long
PATH with sparse closures (tree-like -> no synchronization rigidity). Two things matter for recovering the
discrete quadrant field with SIZE-INDEPENDENT error:
  (1) a DISCRETE solver that uses cycle redundancy (iterated weighted max-vote / max-product BP), not spectral;
  (2) enough CLOSURE DENSITY to make the graph rigid (tree -> errors accumulate; 2D-lattice-like -> Potts
      long-range order -> bounded). DFS explore-once is nearly a tree; re-walking K times adds closures.

This script measures the foundations and tests recovery:
  - chain per-step relative-quadrant accuracy (should be ~100%; its rare errors are what must be corrected)
  - matcher closure accuracy (~95%)
  - ICM/max-vote recovery error vs size, for {explore-1x, dense-3x} x {oracle-closure, matcher-closure}
If ORACLE-dense is FLAT with size => the discrete field IS size-invariantly recoverable given density => the
'grows with size' verdict was a sparse-graph/continuous-approx artifact => GAUGE (beyond global 2-bit) BREAKS
with coverage. If even oracle-dense grows => genuinely gauge-bound (tree-like even when dense).
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
from heading_attack.b03d_integrated import train_matcher, matcher_relquad
HALF = math.pi / 2


def drive(n, seed, revisit=1):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
    order = []
    for rep in range(revisit):
        order += (walk[1:] if rep % 2 == 0 else walk[1:][::-1])
    rec = []; prev_turn = 0.0
    for (cx, cy) in order:
        wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(60):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            gh = (-_grid_heading(g)) % HALF
            rec.append((gh, prev_turn, (int(env.px), int(env.py)), wrap(env.ang - ang0), g, l))
            prev_turn = turn; motion(env, a)
    return rec


def build_edges(rec, matcher, oracle, n_close_per_cell=8):
    """edges as (i, j, r) meaning q_i - q_j = r (mod 4). chain + same-cell closures."""
    T = len(rec); gh = np.array([r[0] for r in rec])
    qtrue = np.array([int(round((rec[t][3] - rec[t][0] + rec[0][0]) / HALF)) for t in range(T)])  # rel start
    edges = []; chain_ok = 0
    for t in range(1, T):
        r = int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF)) % 4
        edges.append((t, t - 1, r, 3.0))                       # chain weight 3 (reliable)
        if (qtrue[t] - qtrue[t - 1]) % 4 == r: chain_ok += 1
    bycell = defaultdict(list)
    for t, rr in enumerate(rec): bycell[rr[2]].append(t)
    clo_ok = clo_tot = 0
    for c, ts in bycell.items():
        if len(ts) < 2: continue
        for a in ts[:3]:
            for b in ts[1:1 + n_close_per_cell]:
                if b <= a: continue
                if oracle:
                    r = int(round((rec[b][3] - rec[a][3]) / HALF)) % 4
                else:
                    r = matcher_relquad(matcher, rec[b][4], rec[b][5], rec[a][4], rec[a][5]) % 4  # q_b-q_a
                edges.append((b, a, r, 1.0))
                if (qtrue[b] - qtrue[a]) % 4 == r: clo_ok += 1
                clo_tot += 1
    return edges, qtrue, chain_ok / max(T - 1, 1), clo_ok / max(clo_tot, 1)


def icm(edges, T, init, sweeps=40):
    """iterated weighted max-vote (max-product / ICM) for Z4. nbrs[i] = list of (j, r, w) with q_i ~ q_j + r."""
    nbr = defaultdict(list)
    for (i, j, r, w) in edges:
        nbr[i].append((j, r, w)); nbr[j].append((i, (-r) % 4, w))
    q = init.copy()
    for _ in range(sweeps):
        changed = 0
        for i in range(T):
            if not nbr[i]: continue
            votes = np.zeros(4)
            for (j, r, w) in nbr[i]: votes[(q[j] + r) % 4] += w
            nb = int(np.argmax(votes))
            if nb != q[i]: q[i] = nb; changed += 1
        if changed == 0: break
    return q


def chain_init(edges, T):
    q = np.zeros(T, int)
    rmap = {}
    for (i, j, r, w) in edges:
        if i == j + 1: rmap[i] = r
    for t in range(1, T): q[t] = (q[t - 1] + rmap.get(t, 0)) % 4
    return q


def eval_size(n, n_mazes, seed0, matcher, revisit, oracle):
    es, cain, clacc = [], [], []
    for mi in range(n_mazes):
        rec = drive(n, seed0 + mi, revisit); T = len(rec); gh = np.array([r[0] for r in rec])
        edges, qtrue, ca, cl = build_edges(rec, matcher, oracle)
        q = icm(edges, T, chain_init(edges, T))
        # reference to start (global offset gauge): align q[0] and gh[0]
        est = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
        tru = np.array([r[3] for r in rec])
        err = np.mean([abs(wrap(est[t] - tru[t])) for t in range(T)]) * 180 / math.pi
        es.append(err); cain.append(ca); clacc.append(cl)
    return np.mean(es), np.mean(cain), np.mean(clacc)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--mazes", type=int, default=2)
    a = ap.parse_args()
    print("Training matcher..."); m = train_matcher(epochs=1200)
    print("b11: discrete Z4 quadrant recovery (ICM max-vote). abs heading err (deg) vs size.")
    print("chain-acc & clo-acc are edge accuracies. Compare explore(1x) vs dense(3x), oracle vs matcher closures.")
    print(f"{'n':>4} {'grid':>9} | {'O-1x':>7} {'O-3x':>7} {'M-1x':>7} {'M-3x':>7} | {'chainA':>7} {'cloA(M,3x)':>10}")
    for n in a.sizes:
        o1, ca, _ = eval_size(n, a.mazes, 5000, m, 1, True)
        o3, _, _ = eval_size(n, a.mazes, 5000, m, 3, True)
        m1, _, _ = eval_size(n, a.mazes, 5000, m, 1, False)
        m3, _, cl3 = eval_size(n, a.mazes, 5000, m, 3, False)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {o1:7.1f} {o3:7.1f} {m1:7.1f} {m3:7.1f} | {ca*100:6.1f}% {cl3*100:9.1f}%")
    print("\nO-3x FLAT => discrete field recoverable size-invariantly given density => gauge breaks with coverage.")
    print("O-3x grows => genuinely gauge-bound. M vs O gap => matcher-noise cost.")
