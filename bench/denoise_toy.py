"""Toy: does DENOISING the latent during training make a recurrent tracker self-correcting (bounded error
on long OOD), WITHOUT imposing any torus? Bounded (non-periodic) arena, so the right attractor topology is
a sheet -- we let it emerge, we don't impose it.

Setup: agent random-walks a bounded 2D arena. Inputs = noisy velocity (odometry) + an INDIRECT position
observation (distances to B fixed beacons + noise). A GRUCell integrates; readout = position.
  - VANILLA: train normally. In-distribution (SHORT trajectories) integration suffices -> it over-relies on
    odometry -> on a LONG OOD trajectory it DRIFTS.
  - DENOISE: during training, randomly mask/perturb the hidden state each step -> the integration is made
    unreliable -> the model is FORCED to lean on the beacon observation to recover -> contractive dynamics
    -> bounded error even on the long OOD trajectory.
Train T=30, test T=300 (10x). If denoise stays bounded while vanilla drifts, the user's idea works.
"""
import numpy as np, torch, torch.nn as nn

B_BEACON = 6
def beacons(rng): return rng.random((B_BEACON, 2)).astype(np.float32)

def gen(n, T, bec, rng, vel_noise=0.04, obs_noise=0.04, R=0.22):
    pos = rng.random((n, 2)).astype(np.float32); P = np.zeros((n, T, 2), np.float32)
    V = np.zeros((n, T, 2), np.float32); O = np.zeros((n, T, 2*B_BEACON), np.float32)   # [masked dist, visibility]
    vel = np.zeros((n, 2), np.float32)
    for t in range(T):
        vel = 0.8*vel + 0.2*rng.normal(0, 0.06, (n, 2))
        pos = np.clip(pos + vel, 0, 1)
        P[:, t] = pos; V[:, t] = vel + rng.normal(0, vel_noise, (n, 2))      # noisy odometry
        d = np.linalg.norm(pos[:, None, :] - bec[None], axis=2)              # (n,B)
        vis = (d < R).astype(np.float32)                                     # PARTIAL: beacon visible only if near
        O[:, t] = np.concatenate([(d + rng.normal(0, obs_noise, d.shape)) * vis, vis], axis=1)
    return V, O, P

class Tracker(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.cell = nn.GRUCell(din, h); self.out = nn.Linear(h, 2); self.H = h
    def forward(self, x, mask_p=0.0, mask_noise=0.0):
        B, T, _ = x.shape; h = torch.zeros(B, self.H, device=x.device); outs = []
        for t in range(T):
            h = self.cell(x[:, t], h); outs.append(self.out(h))
            if self.training and mask_p > 0:                   # zero-mask hidden dims -> must recover next steps
                h = h * (torch.rand_like(h) > mask_p).float()
            if self.training and mask_noise > 0:
                h = h + torch.randn_like(h) * mask_noise
        return torch.stack(outs, 1)

def train(model, x, y, mask_p, mask_noise, epochs=120, bs=64, lr=2e-3, dev="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr); mse = nn.MSELoss(); model.to(dev); x, y = x.to(dev), y.to(dev)
    E = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); mse(model(x[b], mask_p, mask_noise), y[b]).backward(); opt.step()
    return model

def run():
    torch.manual_seed(0); rng = np.random.default_rng(0); bec = beacons(rng)
    Vtr, Otr, Ptr = gen(3000, 30, bec, rng)                   # train: SHORT
    xtr = torch.tensor(np.concatenate([Vtr, Otr], -1)); ytr = torch.tensor(Ptr); din = xtr.shape[2]
    Vte, Ote, Pte = gen(200, 300, bec, rng)                   # test: 10x LONG (OOD horizon)
    xte = torch.tensor(np.concatenate([Vte, Ote], -1)); yte = Pte
    print("error vs time on the 10x-long OOD trajectory (binned):  vanilla   vs   denoise(mask hidden)")
    res = {}
    for name, mp, mn in [("vanilla", 0.0, 0.0), ("denoise", 0.35, 0.05)]:
        m = train(Tracker(din), xtr, ytr, mp, mn)
        m.eval()
        with torch.no_grad(): pred = m(xte).numpy()
        err = np.linalg.norm(pred - yte, axis=2)              # (200,300)
        res[name] = err
    T = 300; edges = np.linspace(0, T, 7).astype(int)
    print(f"{'frac':>6} " + "  ".join(f"{n:>9}" for n in res))
    for k in range(6):
        lo, hi = edges[k], edges[k+1]
        print(f"{(hi/T):>6.2f} " + "  ".join(f"{res[n][:, lo:hi].mean():9.3f}" for n in res))
    print(f"\nfinal (last 50): vanilla {res['vanilla'][:,-50:].mean():.3f}  denoise {res['denoise'][:,-50:].mean():.3f}")
    print("denoise << vanilla at long horizon => masking the latent induced bounded-error self-correction.")

if __name__ == "__main__":
    run()
