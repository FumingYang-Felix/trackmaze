"""First learned baseline: a GRU state-tracker for T1 (localization), to exercise the full harness and
produce the headline curves (learned drift · correction · small->large generalization).

Input per step: K ray distances + last action (4) + per-ray landmark one-hot. Landmark ids are bounded
(ambiguity>=1 -> vocab <= V) so they stay in-range when we test on a LARGER maze than we trained on.
Target: egocentric displacement-from-start (path integration). We report the GRU vs the open-loop
dead-reckoner so the *gap* shows whether the model learned to USE landmarks to correct its drift.

Run:  python train_baseline.py          # CPU/MPS, a few minutes
This is the seed of the baseline suite; transformer / SSM / factorized-TEM trackers slot in the same way.
"""
import numpy as np
import torch, torch.nn as nn
from generate import gen_dataset
from metrics import per_step_error, drift_curve, correction_curve, dead_reckon

V = 12  # landmark vocabulary (bounded via ambiguity>=1 so ids transfer across maze sizes)

def encode(ds, use_landmarks=True):
    feat = torch.from_numpy(ds["feat"])                       # (E,T,K+4): ray distances + last action
    lm = ds["ray_lm"]; E,T,K = lm.shape
    oh = np.zeros((E,T,K,V+1), dtype=np.float32)
    if use_landmarks:                                         # ablate landmarks -> all-zero (rays+action only)
        idx = np.clip(lm+1, 0, V).astype(np.int64)            # -1(none)->0 ; id->id+1 ; clip to V
        np.put_along_axis(oh, idx[...,None], 1.0, axis=3)
    oh = oh.reshape(E,T,K*(V+1))
    x = torch.cat([feat, torch.from_numpy(oh)], dim=2)
    y = torch.from_numpy(ds["disp"])
    return x, y

class GRUTracker(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.gru = nn.GRU(din, h, batch_first=True); self.head = nn.Linear(h, 2)
    def forward(self, x): o,_ = self.gru(x); return self.head(o)

def train(model, x, y, epochs=60, lr=1e-3, bs=32, dev="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    model.to(dev); x = x.to(dev); y = y.to(dev); E = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E)
        tot = 0.0
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); l = lossf(model(x[b]), y[b]); l.backward(); opt.step(); tot += l.item()
        if (ep+1) % 15 == 0: print(f"    epoch {ep+1:3d}  train MSE {tot/((E+bs-1)//bs):.4f}")
    return model

@torch.no_grad()
def gru_errs(model, ds, dev, use_landmarks=True):
    x,_ = encode(ds, use_landmarks); model.eval()
    pred = model(x.to(dev)).cpu().numpy()
    return per_step_error(pred, ds["disp"])

def show(tag, errs, ds, nbins=8):
    dc = drift_curve(errs, nbins)
    cc = {g:v for g,v,_ in correction_curve(errs, ds["reanchor"], 30)}
    print(f"  {tag} drift:      " + " ".join(f"{s}:{v:.2f}" for s,v in dc))
    pts = [g for g in (0,1,2,5,10,20) if g in cc]
    print(f"  {tag} correction: " + " ".join(f"{g}:{cc[g]:.2f}" for g in pts))

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    AMB = 1   # bounded landmark vocab -> ids transfer small->large
    print(f"device={dev}  ambiguity={AMB} (vocab<= {V})")
    print("generating train data (small maze n=6)...")
    tr = gen_dataset(n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)
    din = encode(tr)[0].shape[2]
    # The ablation that IS the bench's question: does seeing landmarks let the tracker correct its drift?
    models = {}
    for use_lm in (True, False):
        tag = "with-landmarks" if use_lm else "NO-landmarks  "
        print(f"\ntraining GRU [{tag}] ...")
        x, y = encode(tr, use_lm); m = GRUTracker(din); train(m, x, y, epochs=60, dev=dev)
        models[use_lm] = m

    for n, T in [(6,160), (12,360)]:   # in-distribution size, then OOD-larger size
        te = gen_dataset(n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
        ew = gru_errs(models[True], te, dev, True); en = gru_errs(models[False], te, dev, False)
        print(f"\n=== test n={n} (grid {2*n+1}x{2*n+1})  LBINS={te['LBINS']}  "
              f"final drift: with-lm {ew[:,-1].mean():.2f}  no-lm {en[:,-1].mean():.2f} cells ===")
        show("with-lm", ew, te); show("no-lm  ", en, te)
