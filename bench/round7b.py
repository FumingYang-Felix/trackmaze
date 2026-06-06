"""Round 7-B (clean thesis metric): is the learned size-invariant topological map NAVIGATIONALLY FAITHFUL?

Finding an UNKNOWN goal is exploration-dominated (SPL -> 0 with size for ANY method, even a perfect map -- you
must sweep ~O(area) to find a hidden goal), so SPL-to-unknown-goal is the wrong thesis metric. The right one
isolates the MAP's value: once built, does it support near-shortest-path planning between places, size-
invariantly? We build the online motion-constrained map (R6-C, rot-invariant descriptor trained n=6,12),
then for many (start,goal) cell pairs compare the shortest-path distance ON THE LEARNED GRAPH to the TRUE
shortest path. Ratio ~1, size-invariant => the map is navigationally faithful => near-optimal navigation is
possible at any size from a small-trained model.

Reports: path-length ratio (learned/true), reachability (fraction of pairs connected in the learned graph),
and node/cell ratio, vs size. Contrast motion-constrained vs global-appearance node identity.
"""
import argparse, numpy as np, torch
from collections import defaultdict, deque, Counter
from env import TrackMazeEnv
from round3a import cell_graph
from round4b import RotInv, feats, build_pairs, train_encoder, gen_dfs
from round6c import drive_collect


def build_graph(descs, cells, thr, constrained):
    """Online topological map -> (cell_to_nodes, node_nbr adjacency, n_nodes). Same matcher as R6-C."""
    node_desc = []; nbr = defaultdict(set); node_cells = defaultdict(Counter); cur = None; node_of = []
    for t in range(len(descs)):
        if cur is None:
            node_desc.append(descs[t].copy()); cur = 0; node_of.append(0); node_cells[0][cells[t]] += 1; continue
        cand = list(nbr[cur]) if constrained else list(range(len(node_desc)))
        best, bs = None, thr
        for nid in cand:
            s = float(descs[t] @ node_desc[nid] / (np.linalg.norm(node_desc[nid]) + 1e-9))
            if s > bs: bs, best = s, nid
        if best is None:
            best = len(node_desc); node_desc.append(descs[t].copy())
        else:
            node_desc[best] = 0.9 * node_desc[best] + 0.1 * descs[t]
        nbr[cur].add(best); nbr[best].add(cur); node_of.append(best); node_cells[best][cells[t]] += 1; cur = best
    # map each true cell to the node that most represents it (majority)
    cell_node = {}
    for nid, cnt in node_cells.items():
        for cell, _ in cnt.items():
            # assign cell to the node where it is most frequent
            pass
    # pick, per cell, the node that contains it most often
    cell_best = {}
    for nid, cnt in node_cells.items():
        for cell, c in cnt.items():
            if cell not in cell_best or c > cell_best[cell][1]: cell_best[cell] = (nid, c)
    cell_node = {cell: nid for cell, (nid, _) in cell_best.items()}
    return cell_node, nbr, len(node_desc)


def gdist(nbr, s, g):
    if s == g: return 0
    prev = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        for v in nbr[u]:
            if v not in prev:
                if v == g: return prev[u] + 1
                prev[v] = prev[u] + 1; q.append(v)
    return None


def true_dist(adj, s, g):
    prev = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in prev:
                if v == g: return prev[u] + 1
                prev[v] = prev[u] + 1; q.append(v)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--thr", type=float, default=0.55); ap.add_argument("--mazes", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=200); ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    tr6, tr12 = gen_dfs(6, 4, "true", 7000), gen_dfs(12, 3, "true", 7100)
    tr = {k: np.concatenate([tr6[k], tr12[k]]) for k in ("geo", "lm", "cell")}
    tr["ep"] = np.concatenate([tr6["ep"], 1000 + tr12["ep"]])
    enc = train_encoder(RotInv(), feats(tr), build_pairs(tr), epochs=a.epochs)
    print(f"Round 7-B: learned-map navigational fidelity vs size (train enc n=6,12). thr={a.thr}")
    print(f"{'n':>4} {'grid':>8} | {'MOTION ratio':>12} {'reach':>6} {'n/cell':>6} | {'GLOBAL ratio':>12} {'reach':>6}")
    for n in a.sizes:
        mr, mre, mnc, gr, gre = [], [], [], [], []
        for m in range(a.mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=16000 + m, max_steps=10 ** 9); env.reset()
            adj = cell_graph(env.wall, n)
            descs, cells_w = drive_collect(env, n, enc)
            cells = [((cx - 1) // 2, (cy - 1) // 2) for (cx, cy) in cells_w]   # wall coords -> cell indices
            rng = np.random.default_rng(m); uniq = list(set(cells))
            for constrained, ratios, reach, ncc in [(True, mr, mre, mnc), (False, gr, gre, None)]:
                cell_node, nbr, nn = build_graph(descs, cells, a.thr, constrained)
                if ncc is not None: ncc.append(nn / len(uniq))
                rs, ok = [], 0; tries = 0
                while len(rs) + (tries - len(rs)) < a.pairs and tries < a.pairs * 4:
                    tries += 1
                    s, g = uniq[rng.integers(len(uniq))], uniq[rng.integers(len(uniq))]
                    if s == g or s not in cell_node or g not in cell_node: continue
                    td = true_dist(adj, s, g)
                    if not td: continue
                    ld = gdist(nbr, cell_node[s], cell_node[g])
                    if ld is None: continue
                    ok += 1; rs.append(ld / td)
                ratios.append(np.mean(rs) if rs else 0); reach.append(ok / max(tries, 1))
        f = lambda x: np.mean(x) if x else 0.0
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {f(mr):12.3f} {f(mre):6.2f} {f(mnc):6.2f} | {f(gr):12.3f} {f(gre):6.2f}")
    print("\nMOTION ratio ~1 & flat + high reach => learned map is navigationally faithful & size-invariant.")
