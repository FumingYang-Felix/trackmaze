"""Round 12 (challenge option 2: FEWER STEPS to the exit). The DFS explorers build a TREE (full edge-by-edge
backtracking -> many steps). To cut steps we need (a) FRONTIER exploration with shortest-known-path routing
(no redundant backtrack), and (b) reliable LOOP CLOSURE so the graph has the maze's shortcuts -> routing is
short. Loop closure under aliasing is the R4-C wall; here we make it CONSERVATIVE: accept a revisit only if
the position estimate is within a tight radius AND the open-exit pattern matches (geometry + structure), so
false merges (which break completeness) are rare. Estimate source = tree(no closure) / odo / mgm / perfect.

Metric: total steps to reach the exit vs size, relative to PERFECT (oracle graph) frontier nav. tree >> others
shows shortcuts matter; mgm vs odo shows whether the accurate learned tracker closes more (longer) loops ->
fewer steps. Low-level moves oracle (like R11); the DECISION (which exit, routing, loop closure) is the test.
"""
import argparse, math, numpy as np, torch
from collections import deque
from env import TrackMazeEnv
from round3a import cell_graph, wrap, motion
from arch_mgm import build
from round11_nav import Tracker, cidx, drive, DIRS


def opens_of(adj, cell):
    return set((dx, dy) for (dx, dy) in DIRS if (cell[0] + dx, cell[1] + dy) in adj.get(cell, []))


def nearest_frontier(nodes, cur):
    """BFS on the learned graph (with loop-edge shortcuts) -> path (list of node ids) to nearest node that
    still has an unexplored open exit."""
    prev = {cur: None}; q = deque([cur])
    while q:
        u = q.popleft()
        if nodes[u]["opens"] - nodes[u]["explored"]:
            path = []; v = u
            while v is not None: path.append(v); v = prev[v]
            return u, path[::-1]
        for d, nb in nodes[u]["nbr"].items():
            if nb not in prev: prev[nb] = u; q.append(nb)
    return None, None


def frontier_nav(env, n, ident, model, r=0.6):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    tr = Tracker(env, "perfect" if ident in ("perfect", "tree") else ident, model)
    nodes = []

    def newnode(cell, est):
        nodes.append(dict(repcell=cell, est=est.copy(), opens=opens_of(adj, cell), explored=set(), nbr={}))
        return len(nodes) - 1

    def match(cell, est):
        if ident == "tree":
            return None                                         # no loop closure -> pure spanning tree
        op = opens_of(adj, cell)
        if ident == "perfect":
            for i, nd in enumerate(nodes):
                if nd["repcell"] == cell: return i
            return None
        best, bd = None, r * r
        for i, nd in enumerate(nodes):
            if nd["opens"] != op: continue                      # structure check (junction signature)
            d = est - nd["est"]; dd = float(d @ d)
            if dd < bd: bd, best = dd, i
        return best

    cur = newnode(start, tr.mu); steps = [0]; safety = 80 * n * n
    while cidx(env) != goal and steps[0] < safety:
        node = nodes[cur]; cell = node["repcell"]
        unexp = list(node["opens"] - node["explored"])
        if unexp:
            d = unexp[0]; node["explored"].add(d)
            nxt = (cell[0] + d[0], cell[1] + d[1])
            drive(env, tr, nxt, steps)
            nc = cidx(env); m = match(nc, tr.mu)
            bd = (-d[0], -d[1])
            if m is None:
                nid = newnode(nc, tr.mu); node["nbr"][d] = nid; nodes[nid]["nbr"][bd] = cur
                nodes[nid]["explored"].add(bd); cur = nid
            else:                                               # loop closure -> shortcut edge
                node["nbr"][d] = m; nodes[m]["nbr"][bd] = cur; nodes[m]["explored"].add(bd); cur = m
        else:
            tgt, path = nearest_frontier(nodes, cur)
            if tgt is None: break
            for nb in path[1:]:                                 # walk the shortest graph path to the frontier
                drive(env, tr, nodes[nb]["repcell"], steps); cur = nb
                if cidx(env) == goal: break
    return cidx(env) == goal, steps[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mgm_v2_big_s0.pt"); ap.add_argument("--arch", default="mgm_v2_big")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=6); ap.add_argument("--r", type=float, default=0.6)
    a = ap.parse_args()
    model = build(a.arch); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model.eval()
    idents = ["perfect", "tree", "odo", "mgm"]
    print(f"Round 12 frontier nav: STEPS to exit (fewer=better). ckpt={a.ckpt} r={a.r}")
    print(f"{'n':>4} {'grid':>8} | " + " | ".join(f"{i:>16}" for i in idents))
    print(f"{'':>4} {'':>8} | " + " | ".join(f"{'succ steps ratio':>16}" for _ in idents))
    for n in a.sizes:
        agg = {k: dict(s=[], st=[]) for k in idents}
        for m in range(a.mazes):
            for ident in idents:
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=18000 + m, max_steps=10 ** 9); env.reset()
                ok, st = frontier_nav(env, n, ident, model, a.r)
                agg[ident]["s"].append(1 if ok else 0); agg[ident]["st"].append(st if ok else np.nan)
        pst = np.nanmean(agg["perfect"]["st"])
        segs = []
        for k in idents:
            sc = np.mean(agg[k]["s"]); stp = np.nanmean(agg[k]["st"])
            segs.append(f"{sc:4.2f} {stp:6.0f} {stp/pst if pst and stp==stp else 0:5.1f}")
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | " + " | ".join(segs))
    print("\nperfect=oracle-graph frontier (shortcuts). tree=no loop closure (full backtrack, most steps).")
    print("odo/mgm = conservative learned loop closure: do they recover shortcuts -> fewer steps, size-invariantly?")
