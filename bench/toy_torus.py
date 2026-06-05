"""Toy: does a path-integrator's latent organize into a TORUS, and does that buy bounded-error tracking?

Setup: an agent random-walks on a PERIODIC 2D arena (wraps mod 1). A small GRU integrates velocity.
  - TORUS model: target = torus embedding [cos2pix, sin2pix, cos2piy, sin2piy] (periodic place code).
  - FLAT control: target = unwrapped cumulative position (no wrap) -> a sheet, drifts on long trajectories.

We then ask two things:
  (1) TOPOLOGY (persistent homology / ripser): is the latent a torus? torus => Betti b1 = 2 (two
      independent loops); a flat sheet => b1 = 0. This is the 'verify topology' step.
  (2) PAYOFF (NOT by construction): on a trajectory 10x LONGER than training, does the torus model keep
      bounded place-error (returning to a place returns the latent to the same state = loop-closure for
      free), while the flat control drifts? Boundedness is not baked into the target.
"""
import numpy as np, torch, torch.nn as nn
from ripser import ripser

def gen_walk(n, T, mom=0.8, sig=0.30, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)); vel = np.zeros((n, 2))
    P = np.zeros((n, T, 2), np.float32); V = np.zeros((n, T, 2), np.float32)
    for t in range(T):
        vel = mom * vel + (1 - mom) * rng.normal(0, sig, (n, 2))
        pos = (pos + vel) % 1.0
        P[:, t] = pos; V[:, t] = vel
    return V, P

def torus_emb(P):
    a = 2 * np.pi * P
    return np.concatenate([np.cos(a[..., :1]), np.sin(a[..., :1]),
                           np.cos(a[..., 1:]), np.sin(a[..., 1:])], -1).astype(np.float32)

def decode_place(pred, periodic):
    if periodic:
        x = np.arctan2(pred[..., 1], pred[..., 0]) / (2*np.pi) % 1.0
        y = np.arctan2(pred[..., 3], pred[..., 2]) / (2*np.pi) % 1.0
        return np.stack([x, y], -1)
    return pred % 1.0

def place_err(pred_place, true_place):
    d = np.abs(pred_place - true_place); d = np.minimum(d, 1 - d)   # circular distance per coord
    return np.linalg.norm(d, axis=-1)

class PIRNN(nn.Module):
    def __init__(self, h=64, periodic=True):
        super().__init__(); self.periodic = periodic; od = 4 if periodic else 2
        self.enc = nn.Linear(od, h); self.gru = nn.GRU(2, h, batch_first=True); self.out = nn.Linear(h, od)
    def forward(self, vel, start):
        h0 = torch.tanh(self.enc(start)).unsqueeze(0)
        o, _ = self.gru(vel, h0)
        return self.out(o), o

def train(model, V, tgt, start, epochs=40, bs=64, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); mse = nn.MSELoss(); E = V.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); pred, _ = model(V[b], start[b]); mse(pred, tgt[b]).backward(); opt.step()
    return model

def betti1(latents, n=800):
    """Count dominant 1-cycles via the largest multiplicative GAP in the sorted lifetime spectrum
    (robust to the finite-sample tail of short noise bars). Returns (#dominant loops, top lifetimes)."""
    idx = np.random.default_rng(0).choice(len(latents), min(n, len(latents)), replace=False)
    pts = latents[idx]; pts = (pts - pts.mean(0)) / (pts.std(0) + 1e-6)
    dg = ripser(pts, maxdim=1)['dgms'][1]
    if len(dg) == 0: return 0, []
    life = np.sort(dg[:, 1] - dg[:, 0])[::-1]
    top = life[:6]
    ratios = top[1:] / np.maximum(top[:-1], 1e-9)        # drop at i => i loops dominate
    k = int(np.argmin(ratios) + 1)
    return k, life[:5].round(2).tolist()

def run(periodic, seed=0, sig=0.12, epochs=120, h=128):
    torch.manual_seed(seed); np.random.seed(seed)
    V, P = gen_walk(2000, 60, sig=sig, seed=1)
    start = torus_emb(P[:, 0]) if periodic else P[:, 0]
    tgt = torus_emb(P) if periodic else (P[:, :1] + np.cumsum(V, 1))   # unwrapped for flat
    m = PIRNN(h=h, periodic=periodic)
    train(m, torch.tensor(V), torch.tensor(tgt), torch.tensor(start), epochs=epochs)
    # topology: latents over a covering set
    with torch.no_grad():
        _, lat = m(torch.tensor(V[:300]), torch.tensor(start[:300]))
    lat = lat.reshape(-1, lat.shape[-1]).numpy()
    b1, life = betti1(lat)
    # payoff: OOD-long trajectory (10x training length)
    Vl, Pl = gen_walk(200, 600, sig=sig, seed=99)
    sl = torus_emb(Pl[:, 0]) if periodic else Pl[:, 0]
    with torch.no_grad():
        pred, _ = m(torch.tensor(Vl), torch.tensor(sl))
    err = place_err(decode_place(pred.numpy(), periodic), Pl)   # (200,600)
    early = err[:, :60].mean(); late = err[:, -60:].mean()
    tag = "TORUS" if periodic else "FLAT "
    print(f"  [{tag}] Betti b1={b1} (top H1 lifetimes {life})  |  place-err early(0-60)={early:.3f} late(540-600)={late:.3f}  drift={late-early:+.3f}")
    return b1, early, late

if __name__ == "__main__":
    print("Verify topology + bounded-error payoff (periodic arena, train T=60, test T=600):")
    run(periodic=True)
    run(periodic=False)
    print("\nExpect: TORUS b1=2 (two loops) & flat error over 10x-long trajectory; FLAT b1=0 & drifting error.")
