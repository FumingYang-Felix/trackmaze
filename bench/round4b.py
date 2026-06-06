"""Round 4-B: can a LEARNED place descriptor, trained on SMALL mazes, push place-recognition far above the
raw view AND transfer OOD to LARGE mazes? R4-A showed raw discriminability is size-invariant (~0.85 AUC,
flat) but too low for high-precision loop closure. If a contrastive encoder trained on 13x13/25x25 reaches
high AUC and keeps it on 41x41..105x105 (unseen sizes), then reliable loop closure is size-invariant and the
remaining global-tracking limit is just the sqrt gauge floor (R3). That is the key enabler for the method.

Two encoders:
  mlp  : MLP on the allo-canonical omni view (needs a heading estimate; here canon='true' = oracle heading)
  rot  : ROTATION-INVARIANT -- a circular-conv over the 32 rays, max-pooled over rotations -> heading-FREE
         place code (recognizes a place regardless of approach heading; sidesteps the heading-drift problem).
Metric: AUC + precision@80%-recall (false-match rate matters for loop closure), per size, OOD.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from round4a import gen_dfs, auc_same_vs_diff


def build_pairs(ds):
    from collections import defaultdict
    g = defaultdict(list)
    for i, (e, c) in enumerate(zip(ds["ep"], ds["cell"])): g[(e, c)].append(i)
    return [v for v in g.values() if len(v) >= 2]


def feats(ds):
    return torch.tensor(np.concatenate([ds["geo"], ds["lm"]], 1), dtype=torch.float32)  # (N,64)


class MLP(nn.Module):
    def __init__(self, din=64, h=128, d=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


class RotInv(nn.Module):
    """Rotation-invariant: treat the 32 rays as a circular signal with 2 channels (geo, lm); circular 1D conv
    (via roll) -> features per rotation -> max-pool over the 32 rotations -> heading-free descriptor."""
    def __init__(self, K=32, c=32, d=64):
        super().__init__(); self.K = K
        self.conv = nn.Conv1d(2, c, 5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(c, c, 5, padding=2, padding_mode="circular")
        self.head = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x):
        b = x.shape[0]; g = x[:, :self.K]; l = x[:, self.K:]
        z = torch.stack([g, l], 1)                       # (b,2,K)
        z = F.relu(self.conv(z)); z = F.relu(self.conv2(z))  # (b,c,K)
        z = z.max(dim=2).values                          # max over ray positions ~ rotation pooling -> invariant
        return F.normalize(self.head(z), dim=-1)


def info_nce(z_a, z_p, tau=0.1):
    logits = z_a @ z_p.t() / tau
    labels = torch.arange(z_a.shape[0])
    return F.cross_entropy(logits, labels)


def train_encoder(enc, X, pairs, epochs=300, bs=256, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    pairs = [p for p in pairs if len(p) >= 2]
    for ep in range(epochs):
        sel = [pairs[k] for k in rng.integers(0, len(pairs), bs)]
        ai = [g[j] for g, j in ((p, rng.integers(len(p))) for p in sel)]
        pj = [g[j] for g, j in ((p, rng.integers(len(p))) for p in sel)]
        za = enc(X[ai]); zp = enc(X[pj])
        loss = info_nce(za, zp)
        opt.zero_grad(); loss.backward(); opt.step()
    return enc


def prec_at_recall(ds, desc, target_recall=0.8, n_pairs=6000, seed=1):
    rng = np.random.default_rng(seed); cell, ep = ds["cell"], ds["ep"]; N = len(cell)
    from collections import defaultdict
    g = defaultdict(list)
    for i in range(N): g[(ep[i], cell[i])].append(i)
    rev = [v for v in g.values() if len(v) >= 2]
    same, diff = [], []
    while len(same) < n_pairs and rev:
        gg = rev[rng.integers(len(rev))]; i, j = rng.choice(len(gg), 2, replace=False)
        same.append(float(desc[gg[i]] @ desc[gg[j]]))
    while len(diff) < len(same) * 5:                         # 5x more negatives (realistic loop-closure imbalance)
        a, b = rng.integers(N), rng.integers(N)
        if cell[a] != cell[b]: diff.append(float(desc[a] @ desc[b]))
    same = np.array(same); diff = np.array(diff)
    thr = np.quantile(same, 1 - target_recall)               # threshold accepting target_recall of true matches
    fp = (diff >= thr).sum(); tp = (same >= thr).sum()
    return tp / max(tp + fp, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default="true", choices=["true", "cmd"])
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    print(f"Round 4-B: learned place descriptor, train on n=6,12, eval OOD. canon={a.canon}")
    tr = {k: np.concatenate([gen_dfs(6, 4, a.canon, 7000)[k], gen_dfs(12, 3, a.canon, 7100)[k]])
          for k in ("geo", "lm", "cell", "ep")}
    # make ep unique across the two pooled sources
    tr["ep"] = np.concatenate([gen_dfs(6, 4, a.canon, 7000)["ep"], 1000 + gen_dfs(12, 3, a.canon, 7100)["ep"]])
    Xtr = feats(tr); pairs = build_pairs(tr)
    print(f"  train nodes={len(tr['cell'])} revisited-cells={len(pairs)}")
    encs = {}
    for name, ctor in [("mlp", MLP), ("rot", RotInv)]:
        encs[name] = train_encoder(ctor(), Xtr, pairs, epochs=a.epochs)

    sizes = [6, 12, 20, 28, 40, 52]
    print(f"\n{'n':>4} {'grid':>8} {'cells':>6} | " + " ".join(f"{'raw':>5} {m+'_AUC':>7} {m+'_P@.8':>7}" for m in ["mlp", "rot"]))
    for n in sizes:
        ev = gen_dfs(n, 3, a.canon, 9000)
        Xev = feats(ev); ncell = len(np.unique(ev["cell"]))
        raw = np.concatenate([ev["geo"], ev["lm"]], 1); raw = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-9, None)
        raw_auc, _ = auc_same_vs_diff(ev, raw.astype(np.float32))
        cells_tag = "*train*" if n in (6, 12) else f"{ncell}"
        parts = [f"{raw_auc:5.2f}"]
        line = f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} {cells_tag:>6} | "
        segs = []
        for m in ["mlp", "rot"]:
            with torch.no_grad(): d = encs[m](Xev).numpy()
            auc, _ = auc_same_vs_diff(ev, d); p = prec_at_recall(ev, d)
            segs.append(f"{raw_auc:5.2f} {auc:7.3f} {p:7.3f}")
        print(line + " ".join(segs))
    print("\n(raw repeated per block for reference; AUC=same-vs-diff, P@.8=precision at 80% recall w/ 5x negatives)")
