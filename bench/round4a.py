"""Round 4-A: the REAL open wall isolated -- can a place descriptor tell "same place" from "different place"
from observations, trained small and generalizing to LARGE mazes (where aliasing grows because there are
more cells)? R3 cheated this with oracle place-identity; R2-B failed at it with hand-coded proximity. If
discrimination (AUC) collapses with size, loop closure can't scale and nothing downstream works.

This file measures RAW (no-training) discriminability of candidate descriptors of the allo-canonical omni
view, as a function of maze size, to find the fundamental aliasing wall before investing in a learned model:
  lm_hist  : landmark-id histogram (vocabulary memorization baseline -- expected to alias OOD)
  omni     : the allo-canonical 32-ray geometry+landmark view at the step (route/heading-invariant, LOCAL)
  omni+ctx : omni view AUGMENTED with a short route context (mean of the last W canon views) -- tests whether
             a little temporal context restores distinctiveness when the single local view aliases.
AUC = P(sim(same-cell pair) > sim(different-cell pair)). 0.5 = chance, 1.0 = perfect.
"""
import argparse, math, numpy as np
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, dfs_walk, wrap, motion


def gen_dfs(n, n_mazes=3, canon="true", seed0=7000):
    """Wide-coverage data: drive the ORACLE DFS walk over the whole maze, record the allo-canonical omni
    view + true cell at every cell-VISIT (node). This actually populates thousands of distinct (aliasing)
    cells at large n -- unlike the spin-in-place scripted explorer."""
    CG, CL, CELL, EP = [], [], [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang; cmd = 0.0
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
        last = None

        def rec():
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            h = (env.ang - ang0) if canon == "true" else cmd
            sh = int(round(h / (2 * math.pi) * KO))
            CG.append(np.roll(g, sh)); CL.append(np.roll(l, sh))
            CELL.append(int(env.py) * 100000 + int(env.px)); EP.append(m)

        rec(); last = (int(env.px), int(env.py))
        for (cx, cy) in walk[1:]:
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(40):
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                err = wrap(math.atan2(dy, dx) - env.ang)
                if abs(err) > 0.20:
                    if err > 0: cmd += 0.20; a = 3
                    else:       cmd -= 0.20; a = 2
                else:
                    a = 0
                motion(env, a)
                cell = (int(env.px), int(env.py))
                if cell != last:
                    rec(); last = cell
    G = np.stack(CG); L = np.stack(CL); C = np.array(CELL); E = np.array(EP)
    return dict(geo=G, lm=L, cell=C, ep=E)


def descriptors(ds, kind, W=8):
    geo, lm, ep = ds["geo"], ds["lm"], ds["ep"]                # (N,32),(N,32),(N,)
    if kind == "lm_hist":
        d = lm.copy()                                          # pure landmark-presence pattern (vocabulary)
    elif kind == "omni":
        d = np.concatenate([geo, lm], axis=1)                 # (N,64) local allo view
    else:  # omni+ctx : augment with a short route context (mean of the last W views in this episode)
        base = np.concatenate([geo, lm], axis=1); ctx = np.zeros_like(base)
        for m in np.unique(ep):
            idx = np.where(ep == m)[0]
            for k, t in enumerate(idx):
                lo = max(0, k - W); ctx[t] = base[idx[lo:k + 1]].mean(0)
        d = np.concatenate([base, ctx], axis=1)
    nrm = np.linalg.norm(d, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return (d / nrm).astype(np.float32)                        # L2-normalized -> cosine via dot


def auc_same_vs_diff(ds, desc, n_pairs=6000, seed=0):
    rng = np.random.default_rng(seed)
    cell, ep = ds["cell"], ds["ep"]; N = len(cell)
    # group node indices by (episode, cell) to find revisits (same true place)
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(N): groups[(ep[i], cell[i])].append(i)
    revis = [v for v in groups.values() if len(v) >= 2]
    same = []
    while len(same) < n_pairs and revis:
        g = revis[rng.integers(len(revis))]; i, j = rng.choice(len(g), 2, replace=False)
        same.append(float(desc[g[i]] @ desc[g[j]]))
        if len(same) > 200 and not len(revis): break
    diff = []
    while len(diff) < len(same):
        a, b = rng.integers(N), rng.integers(N)
        if cell[a] != cell[b]: diff.append(float(desc[a] @ desc[b]))
    same = np.array(same); diff = np.array(diff)
    allv = np.concatenate([same, diff]); order = allv.argsort()
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(allv))
    rs = ranks[:len(same)].sum(); ns = len(same); nd = len(diff)
    return (rs - ns * (ns - 1) / 2) / (ns * nd), ns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40, 52])
    ap.add_argument("--canon", default="true", choices=["true", "cmd"])
    a = ap.parse_args()
    kinds = ["lm_hist", "omni", "omni+ctx"]
    print(f"PLACE-RECOGNITION AUC (same-cell vs different-cell) vs size, canon={a.canon}, oracle DFS coverage "
          f"(0.5=chance,1=perfect). cells = distinct cells actually covered => real aliasing pressure.")
    print(f"{'n':>4} {'grid':>8} {'cells':>6} {'nodes':>6} " + " ".join(f"{k:>9}" for k in kinds))
    rows = {k: [] for k in kinds}
    for n in a.sizes:
        ds = gen_dfs(n=n, n_mazes=3, canon=a.canon)
        ncell = len(np.unique(ds["cell"])); nnode = len(ds["cell"])
        out = []
        for k in kinds:
            desc = descriptors(ds, k); auc, npair = auc_same_vs_diff(ds, desc)
            out.append(auc); rows[k].append(auc)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} {ncell:>6} {nnode:>6} " + " ".join(f"{v:9.3f}" for v in out))
    print("\nAUC drop (n_min -> n_max):")
    for k in kinds:
        r = rows[k]; print(f"  {k:>9}: {r[0]:.3f} -> {r[-1]:.3f}  ({r[-1]-r[0]:+.3f})")
