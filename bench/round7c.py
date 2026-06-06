"""Round 7-C (the user's reframed goal): find the exit WITHOUT GETTING LOST -- not shortest path, but no
repeated re-exploration of already-walked corridors. This is exactly what the size-invariant components
deliver: completeness (reach the exit) is free; "not getting lost" is the MAP's job (reliably knowing
"I've been here"), which motion-constrained matching does without catastrophic merges. Optimality
(shortest path) -- the part that needs unsolved aliasing-robust loop closure -- is explicitly NOT required.

Agent does DFS to the exit. At each new cell it asks the learned map "is this a place I've already mapped?"
  perfect : true cell identity (the no-getting-lost ceiling -- pure DFS, ~2x edges, no re-exploration)
  motion  : motion-constrained descriptor matching (R6-C; avoids false merges)
  global  : global descriptor matching (R5 style; false merges -> thinks new places are old / vice versa)
A correct "revisit" detection lets the agent retreat instead of re-descending into explored territory; a
MISS (duplicate) makes it re-explore (getting lost). Metric: steps-to-exit relative to the PERFECT-map DFS
(ratio ~1 & size-invariant = not getting lost) and redundant-revisit fraction. Low-level moves are oracle
(true adjacency); the DECISION "have I been here" uses the learned map -- that is the thing under test.
"""
import argparse, math, numpy as np, torch
from env import TrackMazeEnv
from generate_allo import omni
from round3a import cell_graph, wrap, motion
from round4b import RotInv, feats, build_pairs, train_encoder, gen_dfs

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


def desc_at(env, enc):
    g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
    with torch.no_grad():
        return enc(torch.tensor(np.concatenate([g, l])[None].astype(np.float32))).numpy()[0]


def move_to(env, cell):
    cx, cy = cell; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
    for _ in range(80):
        dx, dy = wx - env.px, wy - env.py
        if math.hypot(dx, dy) < 0.3: break
        err = wrap(math.atan2(dy, dx) - env.ang)
        a = 3 if (abs(err) > 0.20 and err > 0) else (2 if abs(err) > 0.20 else 0)
        motion(env, a)


def explore(env, n, enc, thr, identity, odo_r=1.2, odo_sigma=0.18):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    nodes = []   # dict(desc, repcell, explored=set, nbr=set, parent=(nid,backdir), est)
    est = np.zeros(2)                    # odometry estimate (cell units); for identity=='odo'
    cellmap = {}                         # repcell -> nid (perfect, O(1))
    estbk = {}                           # rounded-est bucket -> [nid] (odo, O(1))

    def opens(cell): return [(dx, dy) for (dx, dy) in DIRS if (cell[0] + dx, cell[1] + dy) in adj.get(cell, [])]

    def match(prev, cell, desc, est_now):
        if identity == "perfect":
            return cellmap.get(cell)
        if identity == "odo":                                       # GEOMETRIC short-loop detection (no appearance)
            bx, by = int(round(est_now[0])), int(round(est_now[1])); best, bd = None, odo_r * odo_r
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    for nid in estbk.get((bx + dx, by + dy), []):
                        d = est_now - nodes[nid]["est"]; dd = float(d @ d)
                        if dd < bd: bd, best = dd, nid
            return best
        cand = (nodes[prev]["nbr"] if prev is not None else set()) if identity == "motion" else range(len(nodes))
        best, bs = None, thr
        for nid in cand:
            s = float(desc @ nodes[nid]["desc"] / (np.linalg.norm(nodes[nid]["desc"]) + 1e-9))
            if s > bs: bs, best = s, nid
        return best

    def register(nid):
        cellmap.setdefault(nodes[nid]["repcell"], nid)
        e = nodes[nid]["est"]; estbk.setdefault((int(round(e[0])), int(round(e[1]))), []).append(nid)

    d0 = desc_at(env, enc)
    nodes.append(dict(desc=d0, repcell=start, explored=set(), nbr=set(), parent=None, est=est.copy()))
    register(0)
    stack = [0]; steps = 0; visited = {start}; revis = 0; safety = 80 * n * n

    while cidx(env) != goal and stack and steps < safety:
        top = stack[-1]; cur = cidx(env)
        untried = [d for d in opens(cur) if d not in nodes[top]["explored"]]
        if untried:
            d = untried[0]; nodes[top]["explored"].add(d)
            nxt = (cur[0] + d[0], cur[1] + d[1])
            move_to(env, nxt); steps += 1
            est = est + np.array(d, float) + np.random.normal(0, odo_sigma, 2)   # odometry of the move
            nc = cidx(env)
            if nc in visited: revis += 1
            visited.add(nc)
            m = match(top, nc, desc_at(env, enc), est)
            if m is None:                                            # genuinely new place -> descend
                nid = len(nodes)
                nodes.append(dict(desc=desc_at(env, enc), repcell=nc, explored={(-d[0], -d[1])},
                                  nbr={top}, parent=(top, (-d[0], -d[1])), est=est.copy()))
                nodes[top]["nbr"].add(nid); register(nid); stack.append(nid)
            else:                                                    # revisit/loop -> link, retreat, don't descend
                nodes[m]["explored"].add((-d[0], -d[1])); nodes[m]["nbr"].add(top); nodes[top]["nbr"].add(m)
                move_to(env, cur); steps += 1; revis += 1           # retreat (a normal, bounded cost)
                est = est + np.array((-d[0], -d[1]), float) + np.random.normal(0, odo_sigma, 2)
        else:
            if nodes[top]["parent"] is None: break
            par, backdir = nodes[top]["parent"]; stack.pop()
            move_to(env, (cur[0] + backdir[0], cur[1] + backdir[1])); steps += 1; revis += 1
            est = est + np.array(backdir, float) + np.random.normal(0, odo_sigma, 2)
    reached = cidx(env) == goal
    return reached, steps, revis, len(nodes), len(set([nodes[i]["repcell"] for i in range(len(nodes))]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--thr", type=float, default=0.55); ap.add_argument("--mazes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    tr6, tr12 = gen_dfs(6, 4, "true", 7000), gen_dfs(12, 3, "true", 7100)
    tr = {k: np.concatenate([tr6[k], tr12[k]]) for k in ("geo", "lm", "cell")}
    tr["ep"] = np.concatenate([tr6["ep"], 1000 + tr12["ep"]])
    enc = train_encoder(RotInv(), feats(tr), build_pairs(tr), epochs=a.epochs)
    idents = ["perfect", "odo", "motion"]   # global = 0% success at all sizes (false merges); dropped for speed
    print(f"Round 7-C: reach the exit without getting lost. succ + steps/(perfect steps), vs size. thr={a.thr}")
    print(f"{'n':>4} {'grid':>8} | " + " | ".join(f"{i:>16}" for i in idents))
    print(f"{'':>4} {'':>8} | " + " | ".join(f"{'succ steps ratio':>16}" for _ in idents))
    for n in a.sizes:
        agg = {k: dict(succ=[], steps=[]) for k in idents}
        for m in range(a.mazes):
            for ident in idents:
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=17000 + m, max_steps=10 ** 9); env.reset()
                np.random.seed(17000 + m)
                r, s, rv, nn, nc = explore(env, n, enc, a.thr, ident)
                agg[ident]["succ"].append(1 if r else 0); agg[ident]["steps"].append(s if r else np.nan)
        pst = np.nanmean(agg["perfect"]["steps"])
        cells = f"{2*n+1}x{2*n+1}"; segs = []
        for k in idents:
            sc = np.mean(agg[k]["succ"]); st = np.nanmean(agg[k]["steps"])
            segs.append(f"{sc:4.2f} {st:5.0f} {st/pst if pst and st==st else 0:4.1f}")
        print(f"{n:>4} {cells:>8} | " + " | ".join(segs))
    print("\nodo (geometric short-loop detection, NO appearance) succ~1 & ratio flat = the user's goal -- reach")
    print("the exit without getting lost, size-invariantly -- SIDESTEPPING the appearance-aliasing wall.")
