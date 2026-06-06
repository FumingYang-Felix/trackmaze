"""Gauge toy: self-supervised metric recovery with NO position labels. The only signals are
  - ODOMETRY (noisy world-frame velocity) -> pins the gauge (scale + direction + start at 0),
  - LOOP-CLOSURE consistency (same true cell, far apart in time -> same estimate; data association from
    GT cells = known loop closures, UNKNOWN metric -> standard SLAM-with-known-association).
No position MSE. The minimal anchor (start=0 + odometry scale) makes the absolute error meaningful, so we
do NOT need to align to GT at eval (the crutch every prior method hides behind). Anti-collapse is automatic:
the odometry loss forces mu to move by the measured velocity, so loop-closure can't collapse it to a point.

Question: does loop-closure (lambda>0) REDUCE the dead-reckoning drift (lambda=0) on a long OOD trajectory,
with NO position GT? If yes, self-supervised metric recovery works (the field's admitted-open core).
"""
import numpy as np, torch, torch.nn as nn

B = 6
def gen(n, T, bec, rng, vel_noise=0.05, obs_noise=0.04, R=0.22):
    pos = rng.random((n, 2)).astype(np.float32); P = np.zeros((n, T, 2), np.float32)
    V = np.zeros((n, T, 2), np.float32); O = np.zeros((n, T, 2*B), np.float32); vel = np.zeros((n, 2), np.float32)
    for t in range(T):
        vel = 0.8*vel + 0.2*rng.normal(0, 0.06, (n, 2)); pos = np.clip(pos + vel, 0, 1)
        P[:, t] = pos; V[:, t] = vel + rng.normal(0, vel_noise, (n, 2))          # noisy WORLD-frame odometry
        d = np.linalg.norm(pos[:, None, :] - bec[None], axis=2); vis = (d < R).astype(np.float32)
        O[:, t] = np.concatenate([(d + rng.normal(0, obs_noise, d.shape)) * vis, vis], axis=1)
    cell = np.floor(P / 0.1).astype(np.int64); cell = cell[..., 1]*100 + cell[..., 0]   # ~10x10 grid cells
    return V, O, P, cell

class Tracker(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.gru = nn.GRU(din, h, batch_first=True); self.out = nn.Linear(h, 2)
    def forward(self, x): o, _ = self.gru(x); return self.out(o)

def loop_consistency(mu, cell, gap=10):
    Bn, T, _ = mu.shape; tot = 0.0; nb = 0; ti = torch.arange(T); far = (ti[:, None]-ti[None, :]).abs() > gap
    for b in range(Bn):
        same = (cell[b][:, None] == cell[b][None, :]) & far
        if same.sum() == 0: continue
        tot = tot + torch.cdist(mu[b], mu[b])[same].mean(); nb += 1
    return tot / max(nb, 1)

def train(model, V, O, cell, lam, epochs=120, bs=64, lr=2e-3):
    x = torch.tensor(np.concatenate([V, O], -1)); Vt = torch.tensor(V); cellt = torch.tensor(cell)
    opt = torch.optim.Adam(model.parameters(), lr); mse = nn.MSELoss(); E = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); mu = model(x[b])
            step = mu[:, 1:] - mu[:, :-1]
            loss = mse(step, Vt[b][:, 1:]) + (mu[:, 0]**2).mean()        # odometry + anchor start=0 (NO pos GT)
            if lam > 0: loss = loss + lam * loop_consistency(mu, cellt[b])
            loss.backward(); opt.step()
    return model

def run():
    torch.manual_seed(0); rng = np.random.default_rng(0); bec = rng.random((B, 2)).astype(np.float32)
    Vtr, Otr, Ptr, Ctr = gen(3000, 40, bec, rng)                         # train SHORT
    Vte, Ote, Pte, Cte = gen(200, 400, bec, rng)                         # test 10x LONG (OOD horizon)
    xte = torch.tensor(np.concatenate([Vte, Ote], -1))
    print("self-supervised (NO position GT) -- abs error vs time on 10x-long OOD:  odometry-only  vs  +loop-closure")
    res = {}
    for name, lam in [("odom-only", 0.0), ("+loop", 1.0)]:
        m = train(Tracker(xte.shape[2]), Vtr, Otr, Ctr, lam); m.eval()
        with torch.no_grad(): pred = m(xte).numpy()
        res[name] = np.linalg.norm(pred - Pte, axis=2)
    T = 400; edges = np.linspace(0, T, 7).astype(int)
    print(f"{'frac':>6} " + "  ".join(f"{n:>11}" for n in res))
    for k in range(6):
        lo, hi = edges[k], edges[k+1]
        print(f"{hi/T:>6.2f} " + "  ".join(f"{res[n][:, lo:hi].mean():11.3f}" for n in res))
    print(f"\nfinal(last 80): odom-only {res['odom-only'][:,-80:].mean():.3f}  +loop {res['+loop'][:,-80:].mean():.3f}")
    print("if +loop << odom-only at the long horizon => self-supervised loop-closure recovered metric (no pos GT, no GT-align).")

if __name__ == "__main__":
    run()
