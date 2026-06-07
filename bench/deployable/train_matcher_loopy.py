"""The bottleneck was the b03c matcher: 95% on DFS data but 33% (≈chance) on the random-walk LOOPY deployment
distribution -- it did not transfer. The BP back-end recovers the quadrant at 100% given CORRECT closures, and
the chain is 99.7% -- so fixing the matcher unlocks the whole SLAM. Retrain the relative-quadrant matcher on the
actual loopy random-walk distribution and measure quadrant accuracy (held-out, OOD size)."""
import sys, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from deployable.seqplace import rw_drive
from generate_allo import KO
DEV = "cpu"


def ccorr(x, y):
    return np.array([np.dot(y, np.roll(x, s)) for s in range(KO)], np.float32)


def pair_feat(g1, l1, g2, l2):
    return np.concatenate([ccorr(g1, g2), ccorr(l1, l2)]).astype(np.float32)


class Matcher(nn.Module):
    def __init__(self, din=2 * KO, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4))
    def forward(self, x): return self.net(x)


def collect_pairs(sizes, n_mazes, seed0, loop, n_pairs, rng):
    """same-cell pairs from loopy random walks; label = relative quadrant (q_now - q_stored) mod 4."""
    visits = []   # (g, l, true_heading) grouped by (maze,cell)
    bycell = defaultdict(list)
    idx = 0
    for si, n in enumerate(sizes):
        for mi in range(n_mazes):
            G, L, CELL, ANG, DR = rw_drive(n, seed0 + si * 1000 + mi, loop)
            for t in range(len(G)):
                bycell[(si, mi, CELL[t][0], CELL[t][1])].append((G[t], L[t], ANG[t]))
    revis = [k for k, v in bycell.items() if len(v) >= 2]
    X, Y = [], []
    HALF = math.pi / 2
    while len(X) < n_pairs and revis:
        vs = bycell[revis[rng.integers(len(revis))]]
        i, j = rng.choice(len(vs), 2, replace=False)
        g1, l1, a1 = vs[i]; g2, l2, a2 = vs[j]
        X.append(pair_feat(g2, l2, g1, l1))                      # feat(now=1 vs stored=2)... match matcher_relquad order
        Y.append(int(round(((a1 - a2 + math.pi) % (2 * math.pi) - math.pi) / HALF)) % 4)
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y, np.int64))


def train_matcher_loopy(loop=0.9, epochs=2500, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = collect_pairs([6, 12], 4, 1000, loop, 16000, rng)
    X, Y = X.to(DEV), Y.to(DEV)
    m = Matcher().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        idx = torch.randint(0, X.shape[0], (256,), device=DEV)
        loss = F.cross_entropy(m(X[idx]), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def relquad(m, g_now, l_now, g_s, l_s):
    feat = pair_feat(g_now, l_now, g_s, l_s)
    with torch.no_grad():
        return int(m(torch.tensor(feat[None]).to(DEV)).argmax(1).item())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32]); a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    print(f"Retraining matcher on LOOPY random-walk data (loop={a.loop})...", flush=True)
    m = train_matcher_loopy(a.loop)
    print(f"relative-quadrant accuracy vs size (held-out loopy mazes). chance=25%, old DFS-matcher was ~33% here.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'quad acc':>9}", flush=True)
    rng = np.random.default_rng(7)
    for n in a.sizes:
        Xe, Ye = collect_pairs([n], 3, 9000, a.loop, 3000, rng)
        with torch.no_grad(): acc = (m(Xe.to(DEV)).argmax(1).cpu() == Ye).float().mean().item()
        print(f"{n:>4} {2*n+1:>5} | {acc*100:8.1f}%", flush=True)
    print("\nhigh & flat => matcher transfers on the deployment distribution => closures correct => SLAM unlocks.", flush=True)
