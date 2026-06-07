"""Deployable method, piece 1: PLACE RECOGNITION under local aliasing, WITHOUT oracle place-id. This is the
real remaining wall (R4-C: single-view appearance tops ~0.65 top-1 and DROPS with size -- the maze is locally
self-similar). The fix is SEQUENCE (SeqSLAM insight): a window of recent rotation-invariant views is far more
distinctive than one. We measure precision/recall of found loop closures vs maze size in LOOPY mazes (the
regime where the global frame is recoverable IF we find enough correct closures).

Rotation-invariant descriptor (heading differs between visits): magnitude spectrum |rfft(g)| + |rfft(l)| (shift
= rotation -> invariant). Single-view vs SEQUENCE (concat over a window of W recent steps). Retrieval: for each
query t, nearest earlier t' (gap>G) by descriptor L2; accept as closure if dist<thr. Precision = frac of
accepted pairs that are truly same cell; recall = frac of true revisits recovered. Also a MOTION-GATE variant
(only consider t' within dead-reckoned radius) -- the R6-C prior.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, wrap, motion


def rw_drive(n, seed, loop, cover_mult=8):
    """Random-walk coverage (traverses loop edges). Records per step: g, l, cell, true heading, dead-reckon xy."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); ncell = len(adj); rng = np.random.default_rng(seed)
    target = max(2, int(cover_mult * ncell * math.log(ncell + 2) / 10))
    G, L, CELL, ANG, DR = [], [], [], [], []
    cur = (0, 0); hops = 0; drx, dry, cmd = 0.0, 0.0, 0.0
    while hops < target:
        nbrs = adj.get(cur, [])
        if not nbrs: break
        nxt = nbrs[rng.integers(len(nbrs))]; wx, wy = 2 * nxt[0] + 1.5, 2 * nxt[1] + 1.5
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            G.append(g); L.append(l); CELL.append(nxt); ANG.append(wrap(env.ang - ang0)); DR.append((drx, dry))
            if a == 2: cmd -= 0.20
            elif a == 3: cmd += 0.20
            else: drx += 0.22 * math.cos(cmd); dry += 0.22 * math.sin(cmd)   # dead-reckon (drifts)
            motion(env, a)
        cur = nxt; hops += 1
    return (np.array(G), np.array(L), CELL, np.array(ANG), np.array(DR))


def rotinv(g, l):
    """Rotation-invariant single-view descriptor: magnitude spectrum of circular signals (shift-invariant)."""
    fg = np.abs(np.fft.rfft(g)); fl = np.abs(np.fft.rfft(l))
    d = np.concatenate([fg, fl]); return d / (np.linalg.norm(d) + 1e-9)


def descriptors(G, L, W):
    """Per-step descriptor. W=1 single view; W>1 sequence = concat of last W rotinv descriptors (approach sig)."""
    base = np.array([rotinv(G[t], L[t]) for t in range(len(G))])
    if W == 1: return base
    T, d = base.shape; seq = np.zeros((T, W * d))
    for t in range(T):
        for k in range(W):
            i = max(0, t - k); seq[t, k * d:(k + 1) * d] = base[i]
    return seq / (np.linalg.norm(seq, axis=1, keepdims=True) + 1e-9)


def eval_placerec(G, L, CELL, DR, W, gap=40, motion_gate=None, thr_pct=5.0, nquery=500, seed=0):
    """SAMPLED queries (nquery) for speed. For each query t, nearest earlier t' (|t-t'|>gap) by descriptor;
    accept if dist below thr_pct percentile of candidate dists. motion_gate=R -> only t' within dead-reckon R.
    Returns precision, recall, #accepted."""
    desc = descriptors(G, L, W); T = len(G)
    cell = np.array([c[0] * 100000 + c[1] for c in CELL])
    if T <= gap + 5: return 0.0, 0.0, 0
    rng = np.random.default_rng(seed)
    qs = rng.choice(np.arange(gap, T), size=min(nquery, T - gap), replace=False)
    cand = []; truth_revisit = 0
    for t in qs:
        lim = t - gap; js = np.arange(0, lim)
        if motion_gate is not None:
            dd = np.hypot(DR[js, 0] - DR[t, 0], DR[js, 1] - DR[t, 1]); js = js[dd < motion_gate]
        if len(js) == 0: continue
        dvec = desc[js] - desc[t]; d2 = np.einsum('ij,ij->i', dvec, dvec)
        jbest = js[int(np.argmin(d2))]; cand.append((int(t), int(jbest), math.sqrt(d2.min())))
        if (cell[:lim] == cell[t]).any(): truth_revisit += 1
    if not cand: return 0.0, 0.0, 0
    thr = np.percentile([c[2] for c in cand], thr_pct)
    tp = fp = 0
    for (t, j, dv) in cand:
        if dv <= thr:
            if cell[t] == cell[j]: tp += 1
            else: fp += 1
    prec = tp / max(tp + fp, 1); rec = tp / max(truth_revisit, 1)
    return prec, rec, tp + fp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--mazes", type=int, default=2); ap.add_argument("--loop", type=float, default=0.9); a = ap.parse_args()
    print(f"seqplace: loop-closure place recognition precision/recall vs size. loop={a.loop}.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'single P/R':>14} {'seqW8 P/R':>14} {'seqW8+gate P/R':>16}", flush=True)
    for n in a.sizes:
        s1p = s1r = s8p = s8r = sgp = sgr = 0.0; cnt = 0
        for mi in range(a.mazes):
            G, L, CELL, ANG, DR = rw_drive(n, 5000 + mi, a.loop)
            if len(G) < 100: continue
            p1, r1, _ = eval_placerec(G, L, CELL, DR, 1)
            p8, r8, _ = eval_placerec(G, L, CELL, DR, 8)
            pg, rg, _ = eval_placerec(G, L, CELL, DR, 8, motion_gate=3.0)
            s1p += p1; s1r += r1; s8p += p8; s8r += r8; sgp += pg; sgr += rg; cnt += 1
        cnt = max(cnt, 1)
        print(f"{n:>4} {2*n+1:>5} | {s1p/cnt*100:5.0f}%/{s1r/cnt*100:3.0f}%   {s8p/cnt*100:5.0f}%/{s8r/cnt*100:3.0f}%   "
              f"{sgp/cnt*100:5.0f}%/{sgr/cnt*100:3.0f}%", flush=True)
    print("\nseq >> single & stays high OOD => sequence breaks the aliasing wall. +gate boosts precision.", flush=True)
