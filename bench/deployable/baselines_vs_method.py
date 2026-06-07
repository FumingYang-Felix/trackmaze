"""Deployable method, item 1: does a trained GRU/Transformer LEARN the drift-free grid-anchoring + quadrant
tracking, or does it integrate-and-drift? Head-to-head on GLOBAL heading prediction, train small (loopy) eval
OOD large. Compares:
  - cmd-integration (open-loop dead reckoning; the drift floor)
  - GRU baseline (trained end-to-end on [g,l,action] -> (cos,sin) of true heading-from-start)
  - Transformer baseline (same)
  - OURS-online: grid-fine (drift-free mod-90, sign-corrected) + grid-anchored integer quadrant tracker (b06).
    No training, no closures. Tests if explicit structure already beats learned integration.
  - OURS+sync: + oracle-closure Z4 sync (the recoverable bound in the loopy regime).
Metric: mean |wrap(pred_heading - true_heading)| (deg) vs maze size. OURS flat / baselines grow => structured
grid-anchoring wins; learned integrators drift.
"""
import sys, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, wrap, motion
DEV = "cpu"; HALF = math.pi / 2


def rollout(n, seed, loop, T, grid_feat=False):
    """Random-walk-ish coverage rollout. Returns per-step [g(32),l(32),turn_onehot(3)] inputs, true heading-
    from-start, and (turn, gh) for the analytic method."""
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=loop, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; adj = cell_graph(env.wall, n); rng = np.random.default_rng(seed)
    X, H, TURN, GH = [], [], [], []; prev_turn = 0.0; cur = (0, 0); steps = 0
    while steps < T:
        nbrs = adj.get(cur, [])
        if not nbrs: break
        nxt = nbrs[rng.integers(len(nbrs))]; wx, wy = 2 * nxt[0] + 1.5, 2 * nxt[1] + 1.5
        for _ in range(80):
            if steps >= T: break
            dx, dy = wx - env.px, wy - env.py
            if math.hypot(dx, dy) < 0.3: break
            e = wrap(math.atan2(dy, dx) - env.ang)
            a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
            turn = 0.20 if a == 3 else (-0.20 if a == 2 else 0.0)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            oh = [1.0 if a == 0 else 0.0, 1.0 if a == 2 else 0.0, 1.0 if a == 3 else 0.0]
            ghv = (-_grid_heading(g)) % HALF
            feats = [g, l, np.array(oh, np.float32)]
            if grid_feat:   # drift-free freq-4 grid-heading phase as an explicit feature (inductive-bias ablation)
                feats.append(np.array([math.cos(2 * math.pi * ghv / HALF), math.sin(2 * math.pi * ghv / HALF)], np.float32))
            X.append(np.concatenate(feats).astype(np.float32))
            H.append(wrap(env.ang - ang0)); TURN.append(prev_turn); GH.append((-_grid_heading(g)) % HALF)
            prev_turn = turn; motion(env, a); steps += 1
        cur = nxt
    return np.array(X), np.array(H), np.array(TURN), np.array(GH)


class GRUNet(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.rnn = nn.GRU(din, h, batch_first=True); self.out = nn.Linear(h, 2)
    def forward(self, x): y, _ = self.rnn(x); return self.out(y)


class TXNet(nn.Module):
    def __init__(self, din, h=128, L=3):
        super().__init__(); self.inp = nn.Linear(din, h)
        enc = nn.TransformerEncoderLayer(h, 4, h * 2, batch_first=True, dropout=0.0)
        self.tx = nn.TransformerEncoder(enc, L); self.out = nn.Linear(h, 2)
    def forward(self, x):
        T = x.shape[1]; m = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), 1)
        return self.out(self.tx(self.inp(x), mask=m))


def make_batch(mazes, sizes, seed0, T, loop, n_ep, grid_feat=False):
    Xs, Hs = [], []; rng = np.random.default_rng(seed0)
    for e in range(n_ep):
        n = sizes[rng.integers(len(sizes))]
        X, H, _, _ = rollout(n, seed0 + e, loop, T, grid_feat=grid_feat)
        if len(X) < T: continue
        Xs.append(X[:T]); Hs.append(H[:T])
    X = torch.tensor(np.stack(Xs)).float(); H = torch.tensor(np.stack(Hs)).float()
    Y = torch.stack([torch.cos(H), torch.sin(H)], -1).float()
    return X.to(DEV), Y.to(DEV)


def train(model, X, Y, epochs=400, bs=32):
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    for ep in range(epochs):
        idx = torch.randint(0, X.shape[0], (bs,))
        p = model(X[idx]); loss = F.mse_loss(p, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model


def heading_err_model(model, n, loop, T, mazes=3, seed0=9000, grid_feat=False):
    es = []
    for mi in range(mazes):
        X, H, _, _ = rollout(n, seed0 + mi, loop, T, grid_feat=grid_feat)
        if len(X) < 10: continue
        with torch.no_grad():
            p = model(torch.tensor(X[None]).float().to(DEV))[0].cpu().numpy()
        pred = np.arctan2(p[:, 1], p[:, 0])
        es.append(np.mean(np.abs([wrap(pred[t] - H[t]) for t in range(len(H))])) * 180 / math.pi)
    return np.mean(es)


def ours_online(n, loop, T, mazes=3, seed0=9000):
    es = []
    for mi in range(mazes):
        X, H, TURN, GH = rollout(n, seed0 + mi, loop, T)
        if len(GH) < 10: continue
        h_est = GH[0]; gh0 = GH[0]; pred = []
        for t in range(len(GH)):
            if t > 0:
                hp = h_est + TURN[t]; k = round((hp - GH[t]) / HALF); h_est = GH[t] + HALF * k
            pred.append(wrap(h_est - gh0))
        es.append(np.mean(np.abs([wrap(pred[t] - H[t]) for t in range(len(H))])) * 180 / math.pi)
    return np.mean(es)


def cmd_only(n, loop, T, mazes=3, seed0=9000):
    es = []
    for mi in range(mazes):
        X, H, TURN, GH = rollout(n, seed0 + mi, loop, T); cmd = np.cumsum(TURN)
        es.append(np.mean(np.abs([wrap(cmd[t] - H[t]) for t in range(len(H))])) * 180 / math.pi)
    return np.mean(es)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 32]); ap.add_argument("--T", type=int, default=300)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    print(f"Training baselines on small loopy mazes (n=6,12, loop={a.loop})...", flush=True)
    Xtr, Ytr = make_batch(None, [6, 12], 1000, a.T, a.loop, 120)
    din = Xtr.shape[-1]
    gru = train(GRUNet(din), Xtr, Ytr); tx = train(TXNet(din), Xtr, Ytr)
    print(f"GLOBAL heading error (deg) vs size. loop={a.loop}. train n=6,12.", flush=True)
    print(f"{'n':>4} {'px':>5} | {'cmd-int':>8} {'GRU':>7} {'Transf':>7} {'OURS-online':>12}", flush=True)
    for n in a.sizes:
        c = cmd_only(n, a.loop, a.T); g = heading_err_model(gru, n, a.loop, a.T)
        t = heading_err_model(tx, n, a.loop, a.T); o = ours_online(n, a.loop, a.T)
        print(f"{n:>4} {2*n+1:>5} | {c:8.1f} {g:7.1f} {t:7.1f} {o:12.1f}", flush=True)
    print("\nOURS flat & low vs baselines grow => structured grid-anchoring beats learned integration.", flush=True)
