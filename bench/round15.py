"""Round 15: systematically attack the fixable Stage-1 sub-items, one by one, on the faithful junction
substrate (heading-drift calibrated to the real MGM). Pluggable junction LoopCloser; measure circle-back
precision + nav success/steps vs size at the MGM operating point. Honest expectation: each lever pushes the
achievable-size CONSTANT; the fundamental walls (sqrt drift, heading-quadrant) remain far out.

Closer interface:
  reset()
  match(end_cell, est_end, degree, in_len, nodes, cur) -> node_id | None   (declare loop closure or new)
  note_edge(a, b, length)                                                   (optional: told of a confirmed edge)
nodes[i] = dict(repcell, est, degree, lens:[corridor lengths], opens:frozenset, nbr:{dir:(jid,len,backdir)})
('repcell' is ground truth used ONLY by the harness for movement + the precision metric; closers must decide
from est/degree/lens/opens/graph -- using est (drifted) + heading-INVARIANT structure, never repcell.)

Items:
  base   : degree + single incoming-length + est gate (the R14 closer = baseline)
  item1  : degree + FULL incident corridor-length MULTISET + branch-pattern + est tiebreak (richer fingerprint)
  (item2 pose-graph and item3 hierarchical memory added next, building on the winner.)
"""
import argparse, math, numpy as np
from collections import deque, Counter
from env import TrackMazeEnv
from round3a import cell_graph, wrap, motion

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MGM_OP = (0.10, 0.04)


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)
def deg(adj, c): return len(adj.get(c, []))
def opensd(adj, c): return frozenset((dx, dy) for (dx, dy) in DIRS if (c[0] + dx, c[1] + dy) in adj.get(c, []))


class Sim:
    def __init__(self, env, start, sigma, srot, rng):
        self.env, self.start, self.sigma, self.srot, self.rng = env, start, sigma, srot, rng
        self.drift = np.zeros(2); self.theta = 0.0; self.steps = 0

    def est(self):
        td = np.array([cidx(self.env)[0] - self.start[0], cidx(self.env)[1] - self.start[1]], float)
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([c * td[0] - s * td[1], s * td[0] + c * td[1]]) + self.drift

    def move(self, target):
        cx, cy = target; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        for _ in range(80):
            dx, dy = wx - self.env.px, wy - self.env.py
            if math.hypot(dx, dy) < 0.3: break
            err = wrap(math.atan2(dy, dx) - self.env.ang)
            a = 3 if (abs(err) > 0.20 and err > 0) else (2 if abs(err) > 0.20 else 0)
            motion(self.env, a)
        if self.sigma > 0: self.drift = self.drift + self.rng.normal(0, self.sigma, 2)
        if self.srot > 0: self.theta = self.theta + self.rng.normal(0, self.srot)
        self.steps += 1


def walk_corridor(adj, sim, jcell, d, goal):
    cur = jcell; prev = None; length = 0; nd = (cur[0] + d[0], cur[1] + d[1]); came = d
    while True:
        sim.move(nd); length += 1; prev = cur; cur = cidx(sim.env)
        if cur == goal: return cur, 'goal', length, sim.est(), came
        od = opensd(adj, cur); nxts = [e for e in od if (cur[0] + e[0], cur[1] + e[1]) != prev]
        if len(nxts) == 0: return cur, 'dead', length, sim.est(), came
        if len(nxts) >= 2 or deg(adj, cur) != 2: return cur, 'junction', length, sim.est(), came
        came = nxts[0]; nd = (cur[0] + nxts[0][0], cur[1] + nxts[0][1])


def _bfs_cells(adj, s, g):
    prev = {s: None}; q = deque([s])
    while q:
        u = q.popleft()
        if u == g: break
        for v in adj.get(u, []):
            if v not in prev: prev[v] = u; q.append(v)
    path = []; u = g
    while u is not None: path.append(u); u = prev.get(u)
    return path[::-1]


def junction_nav(env, n, sigma, srot, rng, closer):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    sim = Sim(env, start, sigma, srot, rng); closer.reset()
    nodes = [dict(repcell=start, est=sim.est(), degree=deg(adj, start), explored=set(),
                  nbr={}, parent=None, lens=[], opens=opensd(adj, start))]
    cur = 0; safety = 200 * n; lc_tp = 0; lc_n = 0

    def back(frm, to):
        for c in _bfs_cells(adj, frm, to)[1:]: sim.move(c)

    while cidx(env) != goal and sim.steps < safety:
        node = nodes[cur]; cell = node["repcell"]
        unexp = [d for d in opensd(adj, cell) if d not in node["explored"]]
        if unexp:
            d = unexp[0]; node["explored"].add(d)
            est_cur = sim.est()                                   # est at cur BEFORE walking (for odometry delta)
            end, kind, length, est_end, came = walk_corridor(adj, sim, cell, d, goal)
            if kind == 'goal': break
            if kind == 'dead': back(end, cell); continue
            m = closer.match(end, est_end, est_cur, deg(adj, end), length, nodes, cur)
            node["lens"].append(length)
            if m is not None and m != cur:
                lc_n += 1; lc_tp += 1 if nodes[m]["repcell"] == end else 0
                node["nbr"][d] = (m, length, came); nodes[m]["explored"].add((-came[0], -came[1]))
                nodes[m]["lens"].append(length); closer.on_close(cur, m, est_end - est_cur); back(end, cell)
            else:
                nid = len(nodes)
                nodes.append(dict(repcell=end, est=est_end, degree=deg(adj, end), explored={(-came[0], -came[1])},
                                  nbr={}, parent=(cur, d), lens=[length], opens=opensd(adj, end)))
                node["nbr"][d] = (nid, length, came); closer.on_new(cur, nid, est_end - est_cur); cur = nid
        else:
            if node["parent"] is None: break
            pj, pd = node["parent"]; back(cell, nodes[pj]["repcell"]); cur = pj
    reached = cidx(env) == goal
    prec = (lc_tp / lc_n) if lc_n else 1.0
    return reached, sim.steps, prec, lc_n


# ---------------- closers ----------------
class Base:
    """R14 baseline: degree + incoming length + est gate."""
    def __init__(self, r=0.9): self.r = r
    def reset(self): pass
    def on_new(self, a, b, delta): pass
    def on_close(self, a, b, delta): pass
    def match(self, end, est, est_cur, degree, in_len, nodes, cur):
        best, bd = None, self.r * self.r
        for i, nd in enumerate(nodes):
            if nd["degree"] != degree: continue
            if nd["lens"] and not any(in_len == L for L in nd["lens"]): continue
            dd = float((est - nd["est"]) @ (est - nd["est"]))
            if dd < bd: bd, best = dd, i
        return best


class Item1:
    """Richer HEADING-INVARIANT junction fingerprint: degree + incident corridor-length multiset; est tiebreak."""
    def __init__(self, r=0.8): self.r = r
    def reset(self): pass
    def on_new(self, a, b, delta): pass
    def on_close(self, a, b, delta): pass
    def match(self, end, est, est_cur, degree, in_len, nodes, cur):
        cands = []
        for i, nd in enumerate(nodes):
            if nd["degree"] != degree: continue
            cl = Counter(nd["lens"])
            if cl and in_len not in cl: continue
            dd = float((est - nd["est"]) @ (est - nd["est"]))
            if dd < self.r * self.r: cands.append((dd / (1 + cl[in_len]), i))
        if not cands: return None
        cands.sort(); return cands[0][1]


class Item2:
    """POSE-GRAPH relaxation: keep odometry edges (est-deltas between junctions) + loop-closure edges
    (confident merges => relative position 0); periodically relax (translation-only weighted-Laplacian
    least-squares) to get DRIFT-REDUCED positions; gate the next merge by the RELAXED predicted position
    (tighter than raw est => fewer false merges => more confident closures: the virtuous cycle). A merge is
    accepted only when CONFIDENT (degree+length compatible AND relaxed-predicted position within a tight gate
    AND clearly best vs 2nd-best) so the pose-graph isn't poisoned."""
    def __init__(self, r=0.7, margin=0.5):
        self.r = r; self.margin = margin
    def reset(self):
        self.pos = {0: np.zeros(2)}; self.odo = []; self.loops = []   # relaxed positions; edges
    def on_new(self, a, b, delta):
        self.pos[b] = self.pos.get(a, np.zeros(2)) + delta; self.odo.append((a, b, np.array(delta)))
    def on_close(self, a, b, delta):
        self.odo.append((a, b, np.array(delta))); self.loops.append((a, b)); self._relax()
    def _relax(self):
        import scipy.sparse as sp, scipy.sparse.linalg as spla
        ids = sorted(self.pos); idx = {k: i for i, k in enumerate(ids)}; N = len(ids)
        if N < 2: return
        L = sp.lil_matrix((N, N)); bx = np.zeros(N); by = np.zeros(N)
        def edge(a, b, dx, dy, w):
            ia, ib = idx[a], idx[b]
            L[ia, ia] += w; L[ib, ib] += w; L[ia, ib] -= w; L[ib, ia] -= w
            bx[ia] -= w * dx; bx[ib] += w * dx; by[ia] -= w * dy; by[ib] += w * dy
        for a, b, d in self.odo:
            if a in idx and b in idx: edge(a, b, d[0], d[1], 1.0)
        for a, b in self.loops:
            if a in idx and b in idx: edge(a, b, 0.0, 0.0, 5.0)         # same place
        anchor = idx[0]; free = [i for i in range(N) if i != anchor]
        if not free: return
        Lf = L.tocsr()[free][:, free]
        xs = np.zeros(N); ys = np.zeros(N)
        try:
            xs[free] = spla.spsolve(Lf, bx[free]); ys[free] = spla.spsolve(Lf, by[free])
        except Exception:
            return
        for k in ids: self.pos[k] = np.array([xs[idx[k]], ys[idx[k]]])
    def match(self, end, est, est_cur, degree, in_len, nodes, cur):
        pred = self.pos.get(cur, np.zeros(2)) + (est - est_cur)         # relaxed cur + observed corridor delta
        cands = []
        for i, nd in enumerate(nodes):
            if i == cur or nd["degree"] != degree: continue
            if nd["lens"] and not any(in_len == L for L in nd["lens"]): continue
            p = self.pos.get(i, nd["est"])
            dd = float((pred - p) @ (pred - p))
            if dd < self.r * self.r: cands.append((dd, i))
        if not cands: return None
        cands.sort()
        if len(cands) >= 2 and cands[1][0] - cands[0][0] < self.margin * self.margin: return None  # ambiguous -> skip
        return cands[0][1]


CLOSERS = {"base": Base, "item1": Item1, "item2": Item2}


def oracle_steps(env, n):
    class Oracle:
        def reset(self): pass
        def on_new(self, a, b, delta): pass
        def on_close(self, a, b, delta): pass
        def match(self, end, est, est_cur, degree, in_len, nodes, cur):
            for i, nd in enumerate(nodes):
                if nd["repcell"] == end: return i
            return None
    ok, st, _, _ = junction_nav(env, n, 0.0, 0.0, np.random.default_rng(0), Oracle())
    return st if ok else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--closers", nargs="+", default=["base", "item1"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40, 52])
    ap.add_argument("--mazes", type=int, default=10); ap.add_argument("--op", type=float, nargs=2, default=list(MGM_OP))
    a = ap.parse_args(); st, sr = a.op
    print(f"Round 15 @ op sigma={st},sigma_rot={sr}: success / steps-vs-oracle / LC-precision, vs size")
    print(f"{'n':>4} {'grid':>8} | " + " | ".join(f"{c:>22}" for c in a.closers))
    for n in a.sizes:
        orc = []
        for m in range(a.mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=18000 + m, max_steps=10 ** 9); env.reset()
            o = oracle_steps(env, n); orc.append(o if o else np.nan)
        om = np.nanmean(orc); seg = []
        for cname in a.closers:
            sc, rt, pr = [], [], []
            for m in range(a.mazes):
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=18000 + m, max_steps=10 ** 9); env.reset()
                ok, steps, prec, lcn = junction_nav(env, n, st, sr, np.random.default_rng(1000 + m), CLOSERS[cname]())
                sc.append(1 if ok else 0); rt.append(steps / om if ok else np.nan); pr.append(prec)
            seg.append(f"{np.mean(sc):4.2f} {np.nanmean(rt):4.1f} p{np.mean(pr):4.2f}")
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | " + " | ".join(seg))
    print("\nLC-precision p = fraction of loop closures that are TRUE (the circle-back accuracy). Higher+flatter")
    print("with size = the lever pushed the achievable range. item1 should beat base; both cap where est-gate binds.")
