"""b10: GAUGE ATTACK via Z4 SYNCHRONIZATION. The big reframe: I declared the quadrant 'gauge-bound, grows
with size' from b05's EFFECTIVE-RESISTANCE floor -- but that models a CONTINUOUS Gaussian variable (2D Gaussian
free field -> log-divergent, genuinely unbounded). The quadrant is DISCRETE (Z4). Recovering it from pairwise
relative measurements is GROUP (angular) SYNCHRONIZATION = a 2D Potts model, which BELOW a noise threshold has
LONG-RANGE ORDER: bounded fluctuations, EXACT recovery up to a single global offset, INDEPENDENT of size. The
continuous floor does NOT bound the discrete field. If true, the quadrant FIELD is size-invariantly recoverable
and only the global 2-bit offset (the 4-fold symmetry, b08-proven irreducible) remains.

Method: nodes = timesteps. Edges:
  - chain (t-1,t): rel quadrant r = round((turn - wrapHALF(gh_t-gh_{t-1}))/HALF)  [drift-free per-step, ~100%]
  - closure (same cell): matcher relative quadrant (95%)  [the long-range cycles that create rigidity]
Build Hermitian H[i,j]=w*omega^{r_ij}, omega=exp(i*2pi/4); leading eigenvector -> z_t -> q_t. Heading est =
grid fine + 90*(q_t - q_0). Compare to chain-integration (the drifting baseline) and to true, vs size.
FLAT with size => the quadrant is recoverable size-invariantly => the 'grows with size' verdict was a
continuous-approx artifact => gauge (beyond the single global 2-bit) is BROKEN.
"""
import sys, os, math, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
from heading_attack.b03d_integrated import train_matcher, matcher_relquad
HALF = math.pi / 2
OM = np.exp(1j * 2 * math.pi / 4)   # 4th root of unity = 1j


def wrapH(x):
    return x - round(x / HALF) * HALF


def drive(n, seed):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang
    adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0; T = 40 * n
    rec = []  # (gh_signcorr, turn_to_this, cell, true_rel, g, l)
    prev_turn = 0.0
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
            gh = (-_grid_heading(g)) % HALF      # sign-corrected, normalized to [0, HALF)
            rec.append((gh, prev_turn, (int(env.px), int(env.py)), wrap(env.ang - ang0), g, l))
            prev_turn = turn
            motion(env, a); steps += 1
    return rec


def sync_quadrant(rec, matcher, w_close=1.0, w_chain=1.0, n_close_per_cell=6):
    T = len(rec); gh = np.array([r[0] for r in rec])
    rows, cols, vals = [], [], []
    def add(i, j, r, w):
        rows.append(i); cols.append(j); vals.append(w * OM ** (r % 4))
        rows.append(j); cols.append(i); vals.append(w * OM ** ((-r) % 4))
    for t in range(1, T):
        r = int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF))   # q_t - q_{t-1} (RAW gh diff captures crossings)
        add(t, t - 1, r, w_chain)
    bycell = defaultdict(list)
    for t, rr in enumerate(rec): bycell[rr[2]].append(t)
    nclose = 0
    for c, ts in bycell.items():
        if len(ts) < 2: continue
        a = ts[0]
        for b in ts[1:1 + n_close_per_cell]:
            rq = matcher_relquad(matcher, rec[b][4], rec[b][5], rec[a][4], rec[a][5])  # q_b - q_a
            add(b, a, rq, w_close); nclose += 1
    H = sp.csr_matrix((vals, (rows, cols)), shape=(T, T), dtype=complex)
    # leading eigenvector (largest real eigenvalue, H Hermitian)
    try:
        val, vec = spla.eigs(H, k=1, which='LR', maxiter=5000)
        z = vec[:, 0]
    except Exception:
        val, vec = np.linalg.eigh(H.toarray()); z = vec[:, -1]
    q = np.mod(np.round(np.angle(z) / (math.pi / 2)), 4).astype(int)
    return q, nclose


def eval_size(n, n_mazes, seed0, matcher):
    es, ec, nc = [], [], []
    for mi in range(n_mazes):
        rec = drive(n, seed0 + mi); T = len(rec); gh = np.array([r[0] for r in rec])
        true_rel = np.array([r[3] for r in rec])
        q, nclose = sync_quadrant(rec, matcher)
        est = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
        err = np.mean([abs(wrap(est[t] - true_rel[t])) for t in range(T)]) * 180 / math.pi
        # chain-integration baseline (drifting): integrate per-step rel quadrant from start
        qc = np.zeros(T, int)
        for t in range(1, T): qc[t] = qc[t - 1] + int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF))
        estc = np.array([wrap((gh[t] - gh[0]) + HALF * (qc[t] - qc[0])) for t in range(T)])
        errc = np.mean([abs(wrap(estc[t] - true_rel[t])) for t in range(T)]) * 180 / math.pi
        es.append(err); ec.append(errc); nc.append(nclose)
    return np.mean(es), np.mean(ec), int(np.mean(nc))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64, 89])
    ap.add_argument("--mazes", type=int, default=3)
    a = ap.parse_args()
    print("Training matcher (n=6,12)..."); m = train_matcher(epochs=1200)
    print("b10: Z4 SYNCHRONIZATION of the quadrant field. abs heading err (deg), referenced to start. vs size.")
    print(f"{'n':>4} {'grid':>9} | {'SYNC':>8} {'chain-integ':>12} {'#closures':>10}")
    for n in a.sizes:
        es, ec, nc = eval_size(n, a.mazes, 5000, m)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {es:8.1f} {ec:12.1f} {nc:10d}")
    print("\nSYNC flat with size (<< chain-integ) => quadrant field recoverable size-invariantly => gauge broken")
    print("except the single global 2-bit offset (4-fold symmetry, irreducible). chain-integ grows => drift.")
