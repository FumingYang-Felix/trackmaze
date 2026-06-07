"""b03c: LEARNED relative-rotation (quadrant) matcher, to beat the raw cross-corr's 62% quadrant accuracy.

The grid gives the fine heading (mod 90); the only missing bit for a loop closure is the discrete relative
QUADRANT (0/90/180/270) between two same-cell visits. Raw cross-corr gets it ~62% (38% fail because the place
looks ~4-fold-symmetric in geometry). A learned matcher can use the LANDMARK channel + the full correlation
shape (geo and lm) to disambiguate the 4 candidate rotations better -- IF the place has any rotation-breaking
asymmetry. Input = circular cross-correlation vectors (geo and lm) of the two views (KO each) + their raw
overlap stats; output = 4-class relative quadrant. Train on same-cell pairs (oracle id), eval OOD.

If learned >> 62% (and flat with size) -> reliable quadrant closures -> feed a GENTLE re-anchor (avoiding the
b03b greedy failure) -> heading. If it caps near 62% -> rotation-aliasing is fundamental for this maze.
"""
import sys, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def collect(n, n_mazes, seed0, T):
    vis = []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                e = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                vis.append((m, (int(env.px), int(env.py)), wrap(env.ang - ang0), g, l))
                motion(env, a); steps += 1
    return vis


def ccorr(x, y):
    return np.array([np.dot(y, np.roll(x, s)) for s in range(KO)], np.float32)


def pair_feat(v1, v2):
    return np.concatenate([ccorr(v1[3], v2[3]), ccorr(v1[4], v2[4])]).astype(np.float32)  # geo + lm cross-corr (2*KO)


def quad_label(v1, v2):
    return int(round(wrap(v2[2] - v1[2]) / (math.pi / 2))) % 4   # relative quadrant 0..3


def make_pairs(vis, n_pairs, rng, same_frac=0.5):
    bycell = defaultdict(list)
    for v in vis: bycell[(v[0], v[1])].append(v)
    revis = [c for c, vs in bycell.items() if len(vs) >= 2]
    X, Y = [], []
    while len(X) < n_pairs:
        if revis and rng.random() < 0.85:
            vs = bycell[revis[rng.integers(len(revis))]]; i, j = rng.choice(len(vs), 2, replace=False)
            v1, v2 = vs[i], vs[j]
        else:
            v1, v2 = vis[rng.integers(len(vis))], vis[rng.integers(len(vis))]
            if v1[1] != v2[1] or v1[0] != v2[0]: continue
        X.append(pair_feat(v1, v2)); Y.append(quad_label(v1, v2))
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


class Matcher(nn.Module):
    def __init__(self, din=2 * KO, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4))
    def forward(self, x): return self.net(x)


def raw_quad_acc(vis, rng, n=1500):
    """Baseline: raw cross-corr argmax -> nearest-90 quadrant accuracy on same-cell pairs."""
    bycell = defaultdict(list)
    for v in vis: bycell[(v[0], v[1])].append(v)
    revis = [c for c, vs in bycell.items() if len(vs) >= 2]; ok = tot = 0
    for _ in range(n):
        if not revis: break
        vs = bycell[revis[rng.integers(len(revis))]]; i, j = rng.choice(len(vs), 2, replace=False)
        c = ccorr(vs[i][3], vs[j][3]); s = int(np.argmax(c)); rel = 2 * math.pi * (s - KO if s > KO // 2 else s) / KO
        if int(round(rel / (math.pi / 2))) % 4 == quad_label(vs[i], vs[j]): ok += 1
        tot += 1
    return ok / max(tot, 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64]); ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    tr = collect(6, 5, 1000, 240) + collect(12, 4, 1100, 480)
    Xtr, Ytr = make_pairs(tr, 12000, rng)
    Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV)
    m = Matcher().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(a.epochs):
        idx = torch.randint(0, Xtr.shape[0], (256,), device=DEV)
        loss = F.cross_entropy(m(Xtr[idx]), Ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    print(f"b03c learned quadrant matcher (4-class rel-quadrant acc) vs raw cross-corr (62%). train n=6,12.")
    print(f"{'n':>4} {'grid':>8} | {'learned':>8} {'raw xcorr':>9}")
    for n in a.eval_sizes:
        ev = collect(n, 3, 9000, 40 * n); Xe, Ye = make_pairs(ev, 3000, np.random.default_rng(1))
        with torch.no_grad(): acc = (m(Xe.to(DEV)).argmax(1).cpu() == Ye).float().mean().item()
        raw = raw_quad_acc(ev, np.random.default_rng(2))
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {acc*100:7.0f}% {raw*100:8.0f}%")
        if a.out:
            with open(a.out, "a") as f: f.write(f"learned {a.seed} {n} {acc:.4f}\nraw {a.seed} {n} {raw:.4f}\n")
    print("\nlearned >> 62% & flat => quadrant closures reliable -> heading attackable. ~62% => rotation-aliasing wall.")
