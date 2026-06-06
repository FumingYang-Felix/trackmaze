"""Round 14 (硬啃 circle-back via the user's structural insight): scope loop closure to JUNCTIONS, not every
cell. The maze is mostly corridors (degree-2 cells); decisions only happen at junctions (degree!=2). Of the
three cases at a branch — (b) wrong/right branch choice and (c) dead-end -> backtrack — both are
TRAJECTORY-CERTAIN (you know which branches you took / that you hit a dead-end). Only (a) looping back to an
ALREADY-VISITED junction via a new corridor needs circle-back DETECTION. Scoping detection to junctions makes
it far easier: junctions are SPARSE (less aliasing pressure) and DISTINCTIVE (degree + branch pattern), and
the est drifts only over CORRIDOR edges (fewer accumulation points). Compare to the per-cell baseline
(round12_sim TightGate, which falls to ~0.50 success at n40 at the calibrated MGM operating point).

Same simulated tracker as round12_sim (est = rotate(true_disp, theta) + drift, calibrated MGM_OP). Low-level
moves oracle; the DECISION (junction loop closure + frontier routing) is the test. Reports success +
steps/oracle vs size, and junction loop-closure precision, at the MGM op point.
"""
import argparse, math, numpy as np
from collections import deque
from env import TrackMazeEnv
from round3a import cell_graph, wrap, motion

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MGM_OP = (0.10, 0.04)


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)
def deg(adj, c): return len(adj.get(c, []))
def opensd(adj, c): return [(dx, dy) for (dx, dy) in DIRS if (c[0] + dx, c[1] + dy) in adj.get(c, [])]


class Sim:
    """Drives env to adjacent cells (oracle low-level) while maintaining a heading-drifting est + true cell."""
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
    """From junction jcell step in dir d and follow the degree-2 corridor to the next junction or dead-end.
    Returns (end_cell, kind in {'junction','dead','goal'}, length, est_at_end, came_dir_into_end)."""
    cur = jcell; prev = None; length = 0
    nd = (cur[0] + d[0], cur[1] + d[1]); came = d
    while True:
        sim.move(nd); length += 1; prev = cur; cur = cidx(sim.env)
        if cur == goal: return cur, 'goal', length, sim.est(), came
        od = opensd(adj, cur)
        nxts = [e for e in od if (cur[0] + e[0], cur[1] + e[1]) != prev]
        if len(nxts) == 0: return cur, 'dead', length, sim.est(), came       # dead-end (degree 1)
        if len(nxts) >= 2 or deg(adj, cur) != 2: return cur, 'junction', length, sim.est(), came
        came = nxts[0]; nd = (cur[0] + nxts[0][0], cur[1] + nxts[0][1])        # corridor continues


def junction_nav(env, n, sigma, srot, rng, gate_r=0.9, use_loopclose=True):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    sim = Sim(env, start, sigma, srot, rng)
    # nodes = junctions; node: dict(repcell, est, degree, explored:set(dir out), nbr:{dir:(jid,len,backdir)}, parent:(jid,backdir))
    nodes = [dict(repcell=start, est=sim.est(), degree=deg(adj, start), explored=set(), nbr={}, parent=None, lens=[])]
    cur = 0; safety = 200 * n; lc_tp = [0]; lc_n = [0]

    def match(cell, est, degree, in_len):
        """Loop closure gated by a HEADING-INVARIANT junction fingerprint: degree must match AND the corridor
        length just traversed (in_len) must match one of the candidate's known incident corridor lengths
        (lengths are step counts -- drift/heading-invariant, genuinely known to the agent). est is only a
        tiebreak among fingerprint-compatible candidates."""
        if not use_loopclose: return None
        cands = []
        for i, nd in enumerate(nodes):
            if nd["degree"] != degree: continue
            if nd["lens"] and not any(abs(in_len - L) <= 0 for L in nd["lens"]): continue   # length fingerprint
            dd = float((est - nd["est"]) @ (est - nd["est"]))
            if dd < gate_r * gate_r: cands.append((dd, i))
        if not cands: return None
        cands.sort(); return cands[0][1]

    while cidx(env) != goal and sim.steps < safety:
        node = nodes[cur]; cell = node["repcell"]
        # branches at this junction not yet explored (and not the corridor we came from -> handled via explored)
        unexp = [d for d in opensd(adj, cell) if d not in node["explored"]]
        if unexp:
            d = unexp[0]; node["explored"].add(d)
            end, kind, length, est_end, came = walk_corridor(adj, sim, cell, d, goal)
            if kind == 'goal': break
            if kind == 'dead':
                # trajectory-certain: walk back to this junction
                # (drive straight back along the corridor by re-targeting the junction cell)
                _back(adj, sim, end, cell)
                continue
            # junction reached: loop closure (case a) or new
            m = match(end, est_end, deg(adj, end), length)
            node["lens"].append(length)
            if m is not None and m != cur:
                lc_n[0] += 1; lc_tp[0] += 1 if nodes[m]["repcell"] == end else 0
                node["nbr"][d] = (m, length, came)
                nodes[m]["explored"].add((-came[0], -came[1])); nodes[m]["lens"].append(length)
                # do NOT descend (it's a known junction); stay to explore other branches here
                _back(adj, sim, end, cell)
            else:
                nid = len(nodes)
                nodes.append(dict(repcell=end, est=est_end, degree=deg(adj, end), explored={(-came[0], -came[1])},
                                  nbr={}, parent=(cur, d), lens=[length]))
                node["nbr"][d] = (nid, length, came); cur = nid
        else:
            if node["parent"] is None: break
            pj, pd = node["parent"]
            # walk back to parent junction along the corridor we came from
            _walk_back_parent(adj, sim, cell, pj, nodes); cur = pj
    reached = cidx(env) == goal
    prec = (lc_tp[0] / lc_n[0]) if lc_n[0] else 1.0
    return reached, sim.steps, prec, lc_n[0]


def _back(adj, sim, frm, to_junction):
    """Drive straight back from a dead-end/known-junction cell to to_junction along the (unique) corridor."""
    path = _bfs_cells(adj, frm, to_junction)
    for c in path[1:]: sim.move(c)


def _walk_back_parent(adj, sim, frm_junction, parent_jid, nodes):
    path = _bfs_cells(adj, frm_junction, nodes[parent_jid]["repcell"])
    for c in path[1:]: sim.move(c)


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


def oracle_steps(env, n):
    ok, st, _, _ = junction_nav(env, n, 0.0, 0.0, np.random.default_rng(0), use_loopclose=True)
    return st if ok else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=8); ap.add_argument("--gate_r", type=float, default=0.9)
    ap.add_argument("--op", type=float, nargs=2, default=list(MGM_OP)); ap.add_argument("--seed0", type=int, default=18000)
    a = ap.parse_args()
    st, sr = a.op
    print(f"Round 14 JUNCTION nav @ op sigma={st},sigma_rot={sr}, gate_r={a.gate_r}: success / steps-vs-oracle / LC-precision")
    print(f"{'n':>4} {'grid':>8} | {'junction':>22} | {'no-loopclose':>14}")
    for n in a.sizes:
        orc = []
        for m in range(a.mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=a.seed0 + m, max_steps=10 ** 9); env.reset()
            o = oracle_steps(env, n); orc.append(o if o else np.nan)
        om = np.nanmean(orc)
        rows = {}
        for tag, lc in [("junction", True), ("noLC", False)]:
            sc, rt, pr = [], [], []
            for m in range(a.mazes):
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=a.seed0 + m, max_steps=10 ** 9); env.reset()
                ok, steps, prec, lcn = junction_nav(env, n, st, sr, np.random.default_rng(1000 + m), a.gate_r, lc)
                sc.append(1 if ok else 0); rt.append(steps / om if ok else np.nan); pr.append(prec)
            rows[tag] = (np.mean(sc), np.nanmean(rt), np.mean(pr))
        j = rows["junction"]; nl = rows["noLC"]
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {j[0]:4.2f} {j[1]:5.1f} prec{j[2]:4.2f} | {nl[0]:4.2f} {nl[1]:5.1f}")
    print("\njunction-scoped loop closure vs per-cell TightGate (round12_sim: 0.50 success @ n40). Higher success")
    print("at scale + LC-precision ~1.0 => the user's junction insight makes circle-back tractable.")
