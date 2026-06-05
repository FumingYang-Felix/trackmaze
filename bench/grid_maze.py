"""Push direction B onto the REAL maze: does shaping the integrator's latent toward a MODULAR (grid)
code help it generalize to LARGER mazes?

Both models are landmark-free path integrators on TrackMaze (input = rays + action; target = egocentric
displacement). The GRID model adds an auxiliary loss: predict a multi-module periodic code of the
displacement, cos/sin(disp . k_m) for wavevectors k_m at several scales/orientations (the grid-cell code
that tiles space periodically). We read displacement from the regression head and compare the OOD-size
drift of GRID vs VANILLA. If the modular code tiles a bigger maze, GRID should extrapolate with less drift.
We also probe whether the latent gains toroidal structure (ripser Betti b1).
"""
import math, numpy as np, torch, torch.nn as nn
from train_eval import cached, AMB, EVAL_SIZES, encode
from metrics import per_step_error

def wavevectors():
    K = []
    for p in (3.0, 5.0, 8.0):                       # spatial periods (cells)
        for th in (0.0, math.pi/3, 2*math.pi/3):    # hex-like orientations
            k = 2*math.pi/p; K.append([k*math.cos(th), k*math.sin(th)])
    return torch.tensor(K, dtype=torch.float32)     # (M,2), M=9

def mod_code(disp, K):
    proj = disp @ K.t()                             # (...,M)
    return torch.cat([torch.cos(proj), torch.sin(proj)], -1)

class GridInt(nn.Module):
    def __init__(self, din, h=128, M=0):
        super().__init__(); self.gru = nn.GRU(din, h, num_layers=2, batch_first=True)
        self.disp = nn.Linear(h, 2); self.code = nn.Linear(h, 2*M) if M else None
    def forward(self, x):
        o, _ = self.gru(x); return self.disp(o), (self.code(o) if self.code is not None else None), o

def feats(ds, use_lm=False): return encode(ds, use_lm)[0]   # full input; use_lm toggles the landmark channels

def train(model, x, y, K, epochs=60, alpha=1.0, bs=32, dev="cpu"):
    opt = torch.optim.Adam(model.parameters(), 1e-3); mse = nn.MSELoss()
    model.to(dev); x, y = x.to(dev), y.to(dev); E = x.shape[0]; tgt = mod_code(y, K.to(dev)) if model.code is not None else None
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            d, c, _ = model(x[b]); loss = mse(d, y[b])
            if c is not None: loss = loss + alpha * mse(c, tgt[b])
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def drift(model, ds, use_lm, dev="cpu"):
    d, _, _ = model(feats(ds, use_lm).to(dev)); return per_step_error(d.cpu().numpy(), ds["disp"])[:, -1].mean()

@torch.no_grad()
def betti1(model, ds, use_lm, dev="cpu", n=700):
    try:
        from ripser import ripser
    except Exception:
        return None
    _, _, lat = model(feats(ds, use_lm).to(dev)); lat = lat.reshape(-1, lat.shape[-1]).cpu().numpy()
    idx = np.random.default_rng(0).choice(len(lat), min(n, len(lat)), replace=False)
    pts = lat[idx]; pts = (pts - pts.mean(0)) / (pts.std(0) + 1e-6)
    dg = ripser(pts, maxdim=1)['dgms'][1]
    if len(dg) == 0: return 0
    life = np.sort(dg[:, 1] - dg[:, 0])[::-1][:6]
    r = life[1:] / np.maximum(life[:-1], 1e-9); return int(np.argmin(r) + 1)

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    K = wavevectors()
    tr = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)
    y = torch.tensor(np.asarray(tr["disp"], np.float32))
    din = feats(tr, True).shape[2]
    # 2x2: {vanilla, grid} x {+landmark, -landmark}. Question: does the grid-structured latent make
    # the LANDMARK BENEFIT (no-lm minus with-lm) flip positive, where the plain tracker's was negative?
    models = {}
    for arch, M in (("vanilla", 0), ("grid", K.shape[0])):
        for use_lm in (True, False):
            torch.manual_seed(0)
            x = feats(tr, use_lm)
            models[(arch, use_lm)] = train(GridInt(din, M=M), x, y, K)
            print(f"trained {arch} {'+lm' if use_lm else '-lm'}")
    EVALS = [(6,144),(12,288),(20,480)]
    print(f"\n{'size':>6} | {'vanilla +lm':>11} {'-lm':>6} {'benefit':>8} | {'grid +lm':>9} {'-lm':>6} {'benefit':>8}")
    for n, T in EVALS:
        te = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
        vw, vo = drift(models[("vanilla", True)], te, True), drift(models[("vanilla", False)], te, False)
        gw, go = drift(models[("grid", True)], te, True), drift(models[("grid", False)], te, False)
        print(f"{('n='+str(n)):>6} | {vw:11.2f} {vo:6.2f} {vo-vw:+8.2f} | {gw:9.2f} {go:6.2f} {go-gw:+8.2f}")
    print("\nbenefit > 0 = landmarks HELP. Question: is grid's benefit less-negative / positive vs vanilla's?")
    te12 = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)
    print(f"latent topology (n=12, +lm): vanilla b1={betti1(models[('vanilla',True)], te12, True)}  "
          f"grid b1={betti1(models[('grid',True)], te12, True)}")
