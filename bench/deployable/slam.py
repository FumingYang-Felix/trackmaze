"""Item (a): the full deployable method = ITERATIVE MOTION-GATED SLAM, no oracle anything. Closes the loop that
(b) showed is needed: appearance place-rec is weak (~0.4 precision) but a TIGHT POSITION GATE makes it reliable;
the position comes from the corrected heading; the heading comes from loop closures; the closures come from the
gate. Bootstrap by iteration:
  0) front-end: grid-fine heading (drift-free mod-90) + chain quadrant (per-step, ~98.5%) -> init heading;
     dead-reckon position with it.
  for it in iters:
    1) propose loop closures: learned-embedding kNN, GATED by current position estimate (radius shrinks each it);
       the b03c matcher gives each accepted pair's relative quadrant.
    2) robust back-end: IRLS/DCS least-squares on the quadrant field (chain + robust closures + anchor) -> q.
    3) re-integrate position with the corrected heading -> tighter gate next iteration.
Output: global heading error vs size in the loopy (recoverable) phase. If it converges flat & low (approaching
the b15/b17 floor) WITHOUT oracle, the deployable method works -- the phase diagram's recoverable region is
realized by an actual estimator. Compare vs OURS-online (no closures) and the open-loop floor.
"""
import sys, os, math, numpy as np, torch
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from generate_allo import omni, _grid_heading
from round3a import cell_graph, wrap, motion
from env import TrackMazeEnv
from heading_attack.b03d_integrated import train_matcher, matcher_relquad
from deployable.learned_placerec import rotinv_cross, Emb, collect_feats, train_emb
HALF = math.pi / 2


def drive(n, seed, loop, cover_mult=8):
    """Random-walk coverage. Per step: gh(sign-corr, [0,HALF)), turn_into, cell(true), true heading, g, l, fwd."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed)
    target = max(2, int(cover_mult * len(adj) * math.log(len(adj) + 2) / 10))
    rec = []; prev_turn = 0.0; cur = (0, 0); hops = 0; moved = 0.0
    while hops < target:
        nbrs = adj.get(cur, [])
        if not nbrs: break
        nxt = nbrs[rng.integers(len(nbrs))]; wx, wy = 2 * nxt[0] + 1.5, 2 * nxt[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            gh = (-_grid_heading(g)) % HALF
            fwd = 0.22 if a == 0 else 0.0
            rec.append(dict(gh=gh, turn=prev_turn, cell=nxt, true=wrap(env.ang - ang0), g=g, l=l, fwd=fwd))
            prev_turn = turn; motion(env, a)
        cur = nxt; hops += 1
    return rec


def chain_quad(rec):
    T = len(rec); gh = np.array([r['gh'] for r in rec]); turn = np.array([r['turn'] for r in rec])
    cr = np.zeros(T, int)
    for t in range(1, T): cr[t] = int(round((turn[t] - (gh[t] - gh[t - 1])) / HALF)) % 4
    return gh, cr


def integrate(cr):
    q = np.zeros(len(cr), int)
    for t in range(1, len(cr)): q[t] = (q[t - 1] + cr[t]) % 4
    return q


def dead_reckon(rec, h):
    """position from forward steps in the estimated heading frame."""
    T = len(rec); pos = np.zeros((T, 2)); x = y = 0.0
    for t in range(T):
        f = rec[t]['fwd']; x += f * math.cos(h[t]); y += f * math.sin(h[t]); pos[t] = (x, y)
    return pos


def robust_quad_LS(gh, cr, closures, iters=4):
    """IRLS continuous LS on the quadrant: chain (q_t-q_{t-1}=cr, w high) + closures (q_a-q_b=rel, DCS robust)
    + anchor q_0=0. Returns integer q (rounded)."""
    T = len(cr); qf = integrate(cr).astype(float)
    wclose = np.ones(len(closures))
    for _ in range(iters):
        rows, cols, vals, b = [], [], [], []; nE = [0]
        def con(i, j, tgt, w):
            rows.extend([nE[0], nE[0]]); cols.extend([i, j]); vals.extend([w, -w]); b.append(w * tgt); nE[0] += 1
        for t in range(1, T): con(t, t - 1, cr[t], 3.0)
        for e, (i, j, rel) in enumerate(closures): con(i, j, rel, 4.0 * wclose[e])
        rows.append(nE[0]); cols.append(0); vals.append(50.0); b.append(0.0); nE[0] += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(nE[0], T))
        qf = spla.spsolve((A.T @ A).tocsc() + 1e-9 * sp.eye(T), A.T @ np.array(b))
        # DCS reweight closures by residual (in quadrant units)
        for e, (i, j, rel) in enumerate(closures):
            r = qf[i] - qf[j] - rel; wclose[e] = (0.7 ** 2 / (0.7 ** 2 + r ** 2))
    return np.round(qf).astype(int) % 4


def propose_closures(rec, Z, pos, matcher, gateR, kth=60.0, gap=40, nq=1500, seed=0, emb_filter=True):
    """For each sampled query, find earlier candidates within the position gate; if emb_filter, keep the
    nearest-embedding one and accept it when its emb-dist is below the kth percentile (looser kth = more
    closures). The matcher gives each accepted pair's relative quadrant. A tight position gate already makes
    candidates high-precision, so accept generously (kth large)."""
    T = len(rec); rng = np.random.default_rng(seed)
    qs = rng.choice(np.arange(gap, T), min(nq, T - gap), replace=False)
    cand = []
    for t in qs:
        js = np.arange(0, t - gap)
        dd = np.hypot(pos[js, 0] - pos[t, 0], pos[js, 1] - pos[t, 1]); js = js[dd < gateR]
        if len(js) == 0: continue
        d2 = ((Z[js] - Z[t]) ** 2).sum(1); jb = int(js[int(np.argmin(d2))]); cand.append((int(t), jb, float(d2.min())))
    if not cand: return []
    thr = np.percentile([c[2] for c in cand], kth) if emb_filter else 1e9
    cl = []
    for (t, j, dv) in cand:
        if dv <= thr:
            rq = matcher_relquad(matcher, rec[t]['g'], rec[t]['l'], rec[j]['g'], rec[j]['l'])  # q_t - q_j
            cl.append((t, j, rq % 4))
    return cl


def run_slam(rec, embf, matcher, iters=3, R0=8.0, oracle_pos=False):
    gh, cr = chain_quad(rec); T = len(rec)
    q = integrate(cr); Z = embf(rec)
    truepos = np.array([[2.0 * r['cell'][0], 2.0 * r['cell'][1]] for r in rec])
    info = []
    for it in range(iters):
        h = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
        if oracle_pos:
            pos = truepos; gateR = 1.5     # oracle-position gate (radius < 1 cell) -> isolates the back-end
        else:
            pos = dead_reckon(rec, np.array([(gh[t] - gh[0]) + HALF * (q[t] - q[0]) for t in range(T)]))
            gateR = R0 * (0.5 ** it)
        cl = propose_closures(rec, Z, pos, matcher, gateR)
        # closure precision (diagnostic, uses true cell)
        if cl:
            prec = np.mean([1.0 if rec[i]['cell'] == rec[j]['cell'] else 0.0 for (i, j, _) in cl])
        else:
            prec = 0.0
        q = robust_quad_LS(gh, cr, cl)
        info.append((len(cl), prec))
    h = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
    true = np.array([r['true'] for r in rec])
    err = np.mean([abs(wrap(h[t] - true[t])) for t in range(T)]) * 180 / math.pi
    return err, info


def online_err(rec):
    gh, cr = chain_quad(rec); T = len(rec); q = integrate(cr)
    h = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
    true = np.array([r['true'] for r in rec])
    return np.mean([abs(wrap(h[t] - true[t])) for t in range(T)]) * 180 / math.pi


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24]); ap.add_argument("--mazes", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    print("Training matcher + place embedding...", flush=True)
    matcher = train_matcher(epochs=1000)
    Xtr, Ctr = collect_feats(6, 4, 1000, a.loop); X2, C2 = collect_feats(12, 3, 1100, a.loop)
    emb = train_emb(np.concatenate([Xtr, X2]), Ctr + C2)
    def embf(rec):
        F_ = np.array([rotinv_cross(r['g'], r['l']) for r in rec], np.float32)
        with torch.no_grad(): return emb(torch.tensor(F_)).cpu().numpy()
    print("ITERATIVE motion-gated SLAM (no oracle). heading err (deg) vs size. loop=%.1f" % a.loop, flush=True)
    print(f"{'n':>4} {'px':>5} | {'online(no-clo)':>14} {'SLAM':>7} {'closure prec/iter':>22}", flush=True)
    for n in a.sizes:
        oe, se, infos = [], [], None
        for mi in range(a.mazes):
            rec = drive(n, 7000 + mi, a.loop)
            oe.append(online_err(rec)); e, info = run_slam(rec, embf, matcher); se.append(e); infos = info
        ip = " ".join(f"{p*100:.0f}%({c})" for c, p in infos)
        print(f"{n:>4} {2*n+1:>5} | {np.mean(oe):14.1f} {np.mean(se):7.1f}   {ip}", flush=True)
    print("\nSLAM < online & flat with size => iterative motion-gated loop closure realizes the recoverable phase", flush=True)
    print("WITHOUT oracle. closure prec rising across iters => the position-gate bootstrap works.", flush=True)
