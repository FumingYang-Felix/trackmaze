"""Round 7: from map to NAVIGATION. The thesis: every component is size-invariant by construction, so an
agent that explores+maps on small mazes navigates large ones near-optimally, where memorizing end-to-end
nets cannot. Task: reach the goal cell from the start in an UNKNOWN maze; metric = SPL-style efficiency
(oracle shortest path / actual path traversed) and redundant-revisit fraction, vs size.

Decision layer (the thesis): DFS exploration using the ONLINE motion-constrained topological map (R6-C,
learned rot-invariant descriptor trained on n=6,12) to decide which corridors are explored and to route back
to the nearest frontier on its OWN graph. Map imperfections (purity ~0.83, duplicate nodes) surface as
navigation inefficiency -- exactly what we measure. Low-level corridor traversal uses the env's true
adjacency (a solved control detail, like R6-C's oracle coverage); the DECISIONS use the learned map.

LEARNED-map (node identity by motion-constrained descriptor) vs PERFECT-map (node identity by true cell) vs
oracle BFS shortest path (SPL 1.0). All coordinates are CELL indices in [0,n); wall coords only for driving.
"""
import argparse, math, numpy as np, torch
from collections import deque
from env import TrackMazeEnv
from generate_allo import omni
from round3a import cell_graph, wrap, motion
from round4b import RotInv, feats, build_pairs, train_encoder, gen_dfs

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


def bfs_path(adj, start, goal):
    prev = {start: None}; q = deque([start])
    while q:
        u = q.popleft()
        if u == goal: break
        for v in adj.get(u, []):
            if v not in prev: prev[v] = u; q.append(v)
    if goal not in prev: return None
    path = []; u = goal
    while u is not None: path.append(u); u = prev[u]
    return path[::-1]


def desc_at(env, enc):
    g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
    with torch.no_grad():
        return enc(torch.tensor(np.concatenate([g, l])[None].astype(np.float32))).numpy()[0]


def step_to_cell(env, target_cell, counter):
    """Low-level oracle execution: move agent to an (adjacent or reachable) target CELL index. counter[0]++ per cell."""
    cx, cy = target_cell; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
    for _ in range(80):
        dx, dy = wx - env.px, wy - env.py
        if math.hypot(dx, dy) < 0.3: break
        err = wrap(math.atan2(dy, dx) - env.ang)
        a = 3 if (abs(err) > 0.20 and err > 0) else (2 if abs(err) > 0.20 else 0)
        motion(env, a)
    counter[0] += 1


def explore_navigate(env, n, enc, thr, use_learned_map):
    adj = cell_graph(env.wall, n)
    start = cidx(env)
    goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    nodes = []          # dict(desc, nbr={dir:nid}, repcell, explored=set(dir))
    visited = [start]; steps = [0]

    def open_dirs(cell):
        return [(dx, dy) for (dx, dy) in DIRS if (cell[0] + dx, cell[1] + dy) in adj.get(cell, [])]

    def match(prev_node, cell, desc):
        if not use_learned_map:
            for nid, nd in enumerate(nodes):
                if nd["repcell"] == cell: return nid
            return None
        if prev_node is None: return None
        for nid in nodes[prev_node]["nbr"].values():
            s = float(desc @ nodes[nid]["desc"] / (np.linalg.norm(nodes[nid]["desc"]) + 1e-9))
            if s > thr: return nid
        return None

    def add_or_get(prev_node, cell, desc, came_dir):
        nid = match(prev_node, cell, desc)
        if nid is None:
            nid = len(nodes); nodes.append(dict(desc=desc.copy(), nbr={}, repcell=cell, explored=set()))
        else:
            nodes[nid]["desc"] = 0.9 * nodes[nid]["desc"] + 0.1 * desc
        if prev_node is not None and came_dir is not None:
            nodes[prev_node]["nbr"][came_dir] = nid
            nodes[nid]["nbr"][(-came_dir[0], -came_dir[1])] = prev_node
            nodes[prev_node]["explored"].add(came_dir)
            nodes[nid]["explored"].add((-came_dir[0], -came_dir[1]))
        return nid

    def node_at(cell, fallback):
        for nid, nd in enumerate(nodes):
            if nd["repcell"] == cell: return nid
        return fallback

    def nearest_frontier(cur_node):
        prev = {cur_node: None}; q = deque([cur_node])
        while q:
            u = q.popleft(); cell = nodes[u]["repcell"]
            if any(d not in nodes[u]["explored"] for d in open_dirs(cell)): return u
            for nb in nodes[u]["nbr"].values():
                if nb not in prev: prev[nb] = u; q.append(nb)
        return None

    cur_node = add_or_get(None, start, desc_at(env, enc), None)
    safety = 60 * n * n
    while cidx(env) != goal and steps[0] < safety:
        cur = cidx(env)
        unexplored = [d for d in open_dirs(cur) if d not in nodes[cur_node]["explored"]]
        if unexplored:
            d = unexplored[0]; nxt = (cur[0] + d[0], cur[1] + d[1])
            step_to_cell(env, nxt, steps); visited.append(cidx(env))
            cur_node = add_or_get(cur_node, cidx(env), desc_at(env, enc), d)
        else:
            tgt = nearest_frontier(cur_node)
            if tgt is None: break
            p = bfs_path(adj, cur, nodes[tgt]["repcell"])
            if not p or len(p) < 2: break
            for c in p[1:]:
                step_to_cell(env, c, steps); visited.append(cidx(env))
                if cidx(env) == goal: break
            cur_node = node_at(cidx(env), cur_node)
    reached = cidx(env) == goal
    oracle = bfs_path(adj, start, goal)
    olen = (len(oracle) - 1) if oracle else None
    redund = 1.0 - len(set(visited)) / max(len(visited), 1)
    return reached, steps[0], olen, redund, len(nodes), len(adj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28])
    ap.add_argument("--thr", type=float, default=0.55); ap.add_argument("--mazes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    tr6, tr12 = gen_dfs(6, 4, "true", 7000), gen_dfs(12, 3, "true", 7100)
    tr = {k: np.concatenate([tr6[k], tr12[k]]) for k in ("geo", "lm", "cell")}
    tr["ep"] = np.concatenate([tr6["ep"], 1000 + tr12["ep"]])
    enc = train_encoder(RotInv(), feats(tr), build_pairs(tr), epochs=a.epochs)
    print(f"Round 7: navigate-to-goal, SPL vs size (train enc n=6,12). thr={a.thr}")
    print(f"{'n':>4} {'grid':>8} | {'LEARN SPL':>9} {'redund':>6} {'succ':>5} | {'PERFECT SPL':>11} {'redund':>6}")
    for n in a.sizes:
        ls, lr, sc, ps, pr = [], [], [], [], []
        for m in range(a.mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=15000 + m, max_steps=10 ** 9); env.reset()
            r1, pl1, ol, rd1, nn1, nc1 = explore_navigate(env, n, enc, a.thr, True)
            env2 = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=15000 + m, max_steps=10 ** 9); env2.reset()
            r2, pl2, ol2, rd2, nn2, nc2 = explore_navigate(env2, n, enc, a.thr, False)
            sc.append(1 if r1 else 0)
            if r1 and ol: ls.append(ol / max(pl1, ol)); lr.append(rd1)
            if r2 and ol2: ps.append(ol2 / max(pl2, ol2)); pr.append(rd2)
        f = lambda x: np.mean(x) if x else 0.0
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {f(ls):9.3f} {f(lr):6.2f} {f(sc):5.2f} | {f(ps):11.3f} {f(pr):6.2f}")
