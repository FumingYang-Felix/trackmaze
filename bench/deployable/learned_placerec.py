"""Item (b): LEARNED contrastive place embedding -- can learning break the ~0.5 precision wall of hand-crafted
descriptors? Rotation-invariant by construction (same-cell visits differ by rotation): features =
[|FFT(g)|, |FFT(l)|, |FFT(g)*conj(FFT(l))|] (power spectra + CROSS-power spectrum; all invariant to a global
ray-roll = rotation, but the cross term keeps the g-l structural relationship the plain power spectrum loses).
Encoder MLP -> L2-normalized embedding, trained with supervised-contrastive (same-cell close, diff-cell far) on
small mazes, eval OOD large loopy mazes.

Measures top-1 loop-closure precision/recall vs size for: hand-crafted |FFT| (baseline) | learned single |
learned + MOTION-GATE (oracle-position radius R). The gate result shows that descriptor+position resolves
aliasing -> motivates the iterative motion-gated SLAM (item a) that bootstraps the position.
"""
import sys, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from deployable.seqplace import rw_drive, rotinv
DEV = "cpu"


def rotinv_cross(g, l):
    """Rotation-invariant feature keeping g-l structure: power spectra + cross-power-spectrum magnitude."""
    G = np.fft.rfft(g); L = np.fft.rfft(l)
    feat = np.concatenate([np.abs(G), np.abs(L), np.abs(G * np.conj(L))]).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-9)


class Emb(nn.Module):
    def __init__(self, din, h=128, d=32):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def collect_feats(n, n_mazes, seed0, loop):
    F_, C_ = [], []
    for mi in range(n_mazes):
        G, L, CELL, ANG, DR = rw_drive(n, seed0 + mi, loop)
        for t in range(len(G)):
            F_.append(rotinv_cross(G[t], L[t])); C_.append((mi, CELL[t][0], CELL[t][1]))
    return np.array(F_, np.float32), C_


def train_emb(Xtr, Ctr, epochs=1500, bs=256, temp=0.2):
    key = np.array([c[0] * 1000000 + c[1] * 1000 + c[2] for c in Ctr])
    X = torch.tensor(Xtr).to(DEV); K = torch.tensor(key).to(DEV)
    m = Emb(Xtr.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        idx = torch.randint(0, X.shape[0], (bs,), device=DEV)
        z = m(X[idx]); kk = K[idx]
        sim = z @ z.t() / temp
        pos = (kk[:, None] == kk[None, :]).float(); pos.fill_diagonal_(0)
        sim.fill_diagonal_(-1e9)
        logp = F.log_softmax(sim, dim=1)
        has = pos.sum(1) > 0
        loss = -(logp * pos).sum(1)[has] / pos.sum(1)[has].clamp(min=1)
        loss = loss.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def precision_recall(emb_fn, n, loop, seed0, n_mazes=3, gate=None, nquery=600, gap=40):
    """emb_fn: feats(NxD)->embeddings(NxK). top-1 earlier match per sampled query; accept top thr_pct=5%.
    gate=R -> restrict to candidates within true-position radius R (oracle gate). Returns prec, rec."""
    P, R = [], []
    for mi in range(n_mazes):
        G, L, CELL, ANG, DR = rw_drive(n, seed0 + mi, loop); T = len(G)
        if T < gap + 50: continue
        Z = emb_fn(G, L); cell = np.array([c[0] * 100000 + c[1] for c in CELL])
        pos = np.array([[2 * c[0], 2 * c[1]] for c in CELL], float)
        rng = np.random.default_rng(0); qs = rng.choice(np.arange(gap, T), min(nquery, T - gap), replace=False)
        cand = []; truth = 0
        for t in qs:
            js = np.arange(0, t - gap)
            if gate is not None:
                dd = np.hypot(pos[js, 0] - pos[t, 0], pos[js, 1] - pos[t, 1]); js = js[dd < gate]
            if len(js) == 0: continue
            d2 = ((Z[js] - Z[t]) ** 2).sum(1); jb = js[int(np.argmin(d2))]
            cand.append((int(t), int(jb), float(d2.min())))
            if (cell[:t - gap] == cell[t]).any(): truth += 1
        if not cand: continue
        thr = np.percentile([c[2] for c in cand], 5.0); tp = fp = 0
        for (t, j, dv) in cand:
            if dv <= thr:
                if cell[t] == cell[j]: tp += 1
                else: fp += 1
        P.append(tp / max(tp + fp, 1)); R.append(tp / max(truth, 1))
    return (np.mean(P) if P else 0.0), (np.mean(R) if R else 0.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32]); a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    print(f"Training learned place embedding (contrastive, n=6,12, loop={a.loop})...", flush=True)
    Xtr, Ctr = collect_feats(6, 4, 1000, a.loop); X2, C2 = collect_feats(12, 3, 1100, a.loop)
    Xtr = np.concatenate([Xtr, X2]); Ctr = Ctr + C2
    m = train_emb(Xtr, Ctr)
    def learned(G, L):
        F_ = np.array([rotinv_cross(G[t], L[t]) for t in range(len(G))], np.float32)
        with torch.no_grad(): return m(torch.tensor(F_).to(DEV)).cpu().numpy()
    def hand(G, L): return np.array([rotinv(G[t], L[t]) for t in range(len(G))], np.float32)
    print(f"top-1 loop-closure precision/recall vs size. loop={a.loop}.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'hand P/R':>11} {'learned P/R':>13} {'learned+gate P/R':>17}", flush=True)
    for n in a.sizes:
        hp, hr = precision_recall(hand, n, a.loop, 9000)
        lp, lr = precision_recall(learned, n, a.loop, 9000)
        gp, gr = precision_recall(learned, n, a.loop, 9000, gate=3.0)
        print(f"{n:>4} {2*n+1:>5} | {hp*100:4.0f}%/{hr*100:3.0f}%   {lp*100:4.0f}%/{lr*100:3.0f}%    {gp*100:4.0f}%/{gr*100:3.0f}%", flush=True)
    print("\nlearned >> hand => learning helps; learned+gate high & flat => descriptor+position resolves aliasing", flush=True)
    print("=> motivates iterative motion-gated SLAM (item a) to bootstrap the position.", flush=True)
