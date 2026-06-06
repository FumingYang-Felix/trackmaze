"""Round 12 substrate: isolate the LOOP-CLOSURE ALGORITHM from the tracker. Frontier exploration to the exit
on an online topological graph; the position estimate is SIMULATED as true-displacement + a random-walk drift
of stdev sigma per move (sigma=0 = perfect/oracle place-id; small sigma ~ a good tracker like MGM; large
sigma ~ raw odometry). A pluggable LoopCloser decides, at each newly-entered cell, whether it is a REVISIT of
an existing graph node (-> add a shortcut edge) or NEW. The whole game: a closer that recovers the maze's
shortcuts (fewer steps, near oracle) WITHOUT false merges (which break completeness), ROBUSTLY as sigma grows
and size grows. Low-level moves are oracle; the decision (loop closure + frontier routing) is the test.

A LoopCloser implements:  reset();  match(cell, est, opens, nodes) -> node_id|None
  cell : current true cell (cx,cy)        est  : (2,) noisy displacement estimate (cell units)
  opens: frozenset of open exit dirs      nodes: list of dicts {repcell, est, opens, nbr:{dir:nid}, hist:[est...]}
Return an existing node id to declare a loop closure (shortcut), or None to create a new node. Closers may
keep their own state and inspect the whole graph (for consistency / multi-hypothesis / pose-graph checks).

run_eval(closer_factory, sigmas, sizes, mazes) -> prints success + steps/oracle vs (sigma,size).
"""
import math, numpy as np
from collections import deque
from env import TrackMazeEnv
from round3a import cell_graph

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)
def opens_of(adj, cell): return frozenset((dx, dy) for (dx, dy) in DIRS if (cell[0] + dx, cell[1] + dy) in adj.get(cell, []))


# ---- baseline closers ----
class NoClose:
    """No loop closure -> pure spanning tree (loops forever in cyclic mazes)."""
    def reset(self): pass
    def match(self, cell, est, opens, nodes): return None


class TightGate:
    """Revisit iff an existing node is within radius r of est AND has the same open-exit pattern."""
    def __init__(self, r=0.6): self.r = r
    def reset(self): pass
    def match(self, cell, est, opens, nodes):
        best, bd = None, self.r * self.r
        for i, nd in enumerate(nodes):
            if nd["opens"] != opens: continue
            d = est - nd["est"]; dd = float(d @ d)
            if dd < bd: bd, best = dd, i
        return best


class Oracle:
    """Perfect place id (sigma-independent) -> the upper bound on shortcut recovery."""
    def reset(self): pass
    def match(self, cell, est, opens, nodes):
        for i, nd in enumerate(nodes):
            if nd["repcell"] == cell: return i
        return None


# ---- frontier navigation harness ----
def _nearest_frontier(nodes, cur):
    prev = {cur: None}; q = deque([cur])
    while q:
        u = q.popleft()
        if nodes[u]["opens"] - nodes[u]["explored"]:
            path = []; v = u
            while v is not None: path.append(v); v = prev[v]
            return path[::-1]
        for d, nb in nodes[u]["nbr"].items():
            if nb not in prev: prev[nb] = u; q.append(nb)
    return None


def navigate(env, n, closer, sigma, rng, sigma_rot=None):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    closer.reset()
    drift = np.zeros(2); theta = 0.0                      # translational drift + accumulated HEADING error
    # heading drift is the dominant failure mode (R3/R6): a real tracker's estimate ROTATES away from truth.
    srot = sigma * 0.4 if sigma_rot is None else sigma_rot

    def est_of():  # simulated tracker: rotate true displacement by accumulated heading error, + transl drift
        td = np.array([cidx(env)[0] - start[0], cidx(env)[1] - start[1]], float)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c * td[0] - s * td[1], s * td[0] + c * td[1]]) + drift

    def step_drift():
        nonlocal drift, theta
        if sigma > 0: drift = drift + rng.normal(0, sigma, 2)
        if srot > 0: theta = theta + rng.normal(0, srot)

    def drive(target):
        cx, cy = target; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
        from round3a import wrap, motion
        for _ in range(80):
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            err = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(err) > 0.20 and err > 0) else (2 if abs(err) > 0.20 else 0)
            motion(env, a)
        step_drift(); steps[0] += 1

    nodes = [dict(repcell=start, est=est_of(), opens=opens_of(adj, start), explored=set(), nbr={}, hist=[est_of()])]
    cur = 0; steps = [0]; safety = 120 * n * n
    while cidx(env) != goal and steps[0] < safety:
        node = nodes[cur]; cell = node["repcell"]
        unexp = list(node["opens"] - node["explored"])
        if unexp:
            d = unexp[0]; node["explored"].add(d); bd = (-d[0], -d[1])
            drive((cell[0] + d[0], cell[1] + d[1]))
            nc = cidx(env); e = est_of(); op = opens_of(adj, nc)
            m = closer.match(nc, e, op, nodes)
            if m is None:
                nid = len(nodes)
                nodes.append(dict(repcell=nc, est=e, opens=op, explored={bd}, nbr={bd: cur}, hist=[e]))
                node["nbr"][d] = nid; cur = nid
            else:
                node["nbr"][d] = m; nodes[m]["nbr"][bd] = cur; nodes[m]["explored"].add(bd)
                nodes[m]["hist"].append(e); cur = m
        else:
            path = _nearest_frontier(nodes, cur)
            if path is None: break
            for nb in path[1:]:
                drive(nodes[nb]["repcell"]); cur = nb
                if cidx(env) == goal: break
    return cidx(env) == goal, steps[0]


def oracle_steps(env, n):
    ok, st = navigate(env, n, Oracle(), 0.0, np.random.default_rng(0)); return st if ok else None


# CALIBRATED operating point: (sigma_trans, sigma_rot) per move that reproduces the real trained MGM nav
# (tight-gate ~100% at small sizes -> ~50% at n40, steps near oracle). Use this to test loop-closure algos.
MGM_OP = (0.10, 0.04)


def run_eval(closer_factory, op_points=(MGM_OP,), sizes=(6, 12, 20, 28, 40), mazes=8, seed0=18000, label=""):
    """For each (sigma_trans, sigma_rot) op point, report (success, mean steps / ORACLE-graph steps) vs size.
    A GOOD closer keeps success ~1.0 AND ratio low (near oracle) as size grows, at the MGM operating point.
    High success + high ratio = wandering; low success = false merges skip the exit. Need BOTH good."""
    print(f"=== {label} : success / (steps / oracle) vs size ===", flush=True)
    print(f"{'n':>4} {'grid':>8} | " + " | ".join(f"s{st},r{sr}".ljust(11) for st, sr in op_points), flush=True)
    out = {}
    for n in sizes:
        orc = []
        for m in range(mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); env.reset()
            ok, st = navigate(env, n, Oracle(), 0.0, np.random.default_rng(m), 0.0); orc.append(st if ok else np.nan)
        orc_mean = np.nanmean(orc); cells = []
        for (strans, srot) in op_points:
            succ, ratio = [], []
            for m in range(mazes):
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); env.reset()
                ok, st = navigate(env, n, closer_factory(), strans, np.random.default_rng(1000 + m), srot)
                succ.append(1 if ok else 0); ratio.append(st / orc_mean if ok else np.nan)
            cells.append((float(np.mean(succ)), float(np.nanmean(ratio)))); out[(n, strans, srot)] = cells[-1]
        seg = " | ".join(f"{sc:3.2f} {r:5.1f}" for sc, r in cells)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {seg}", flush=True)
    return out


def score(out):
    """Single scalar to compare closers: mean over sizes of (success - 0.1*max(0,ratio-1)). Higher=better.
    Rewards completeness; lightly penalizes wandering above oracle. (success is the dominant term.)"""
    return float(np.mean([sc - 0.1 * max(0.0, r - 1.0) for (sc, r) in out.values()]))


if __name__ == "__main__":
    import sys
    fac = {"none": NoClose, "tight": lambda: TightGate(0.6), "oracle": Oracle}
    which = sys.argv[1] if len(sys.argv) > 1 else "tight"
    o = run_eval(fac[which], label=which); print("score:", round(score(o), 3))
