"""Contractive size-invariant tracker -- directly tests the theorem: a local size-invariant recurrent
update generalizes to ARBITRARY maze size iff its error dynamics are CONTRACTIVE (|A|<1).

Mechanism: a perturbation-recovery (contraction) loss. During training, inject a perturbation delta into
the hidden state at a random step tau; the model's SUBSEQUENT outputs must RECOVER to the clean ones
(penalize ||mu_perturbed - mu_clean|| for t>tau, weighted toward later steps -> they must decay to 0).
This trains the recurrent map to pull perturbations (= accumulated drift) back -> |A|<1 -> bounded error
at ANY path length -> size generalization. Train n=6 (13x13), eval at GROWING sizes; flat error-vs-size
curve = the theorem holds. Reports a CONTRACTION diagnostic (how fast injected perturbations decay).
"""
import argparse, numpy as np, torch, torch.nn as nn
from generate_allo import gen_allo
from metrics import per_step_error

def tens(ds):
    geo = torch.from_numpy(ds["canon_geo"]); lm = torch.from_numpy(ds["canon_lm"]); act = torch.from_numpy(ds["action"])
    oh = torch.zeros(*act.shape, 4); oh.scatter_(2, act.unsqueeze(-1), 1.0)
    return torch.cat([geo, lm, oh], -1), torch.from_numpy(ds["disp"])

class CT(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.cell = nn.GRUCell(din, h); self.out = nn.Linear(h, 2); self.H = h
    def forward(self, x, perturb_at=None, delta=None):
        B, T, _ = x.shape; h = torch.zeros(B, self.H, device=x.device); outs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            if perturb_at is not None and t == perturb_at: h = h + delta
            outs.append(self.out(h))
        return torch.stack(outs, 1)

def train(model, x, y, lam, epochs, bs=32, lr=1e-3, dev="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr); mse = nn.MSELoss(); model.to(dev); x, y = x.to(dev), y.to(dev)
    E, T = x.shape[0], x.shape[1]
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); mu = model(x[b]); loss = mse(mu, y[b])
            if lam > 0:                                                    # contraction (perturbation-recovery)
                tau = int(torch.randint(5, T - 20, (1,)))
                delta = torch.randn(len(b), model.H, device=dev) * 0.5
                mup = model(x[b], perturb_at=tau, delta=delta)
                hor = T - tau - 1; w = torch.linspace(0.2, 1.0, hor, device=dev)
                diff = (mup[:, tau+1:] - mu[:, tau+1:].detach()).norm(dim=-1)   # must DECAY over t
                loss = loss + lam * (diff * w).mean()
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def contraction_ratio(model, x, dev):                                     # final/initial perturbation magnitude
    model.eval(); x = x.to(dev); T = x.shape[1]; tau = T // 3
    mu = model(x); delta = torch.randn(x.shape[0], model.H, device=dev) * 0.5
    mup = model(x, perturb_at=tau, delta=delta); d = (mup - mu).norm(dim=-1)   # (B,T)
    init = d[:, tau+1:tau+6].mean().item(); fin = d[:, -20:].mean().item()
    return fin / max(init, 1e-6)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = a.device
    G = dict(ambiguity=1, canon="cmd", rot_noise=0.09, loop=0.3)
    tr = gen_allo(n=6, n_eps=200, T=160, seed=1, **G); x, y = tens(tr); din = x.shape[2]
    model = CT(din); train(model, x, y, a.lam, a.epochs, dev=dev)
    errs = []
    for n in (6, 12, 20, 28, 40):                                         # GROWING size = the OOD test
        te = gen_allo(n=n, n_eps=40, T=24*n, seed=7, **G); xt, _ = tens(te)
        model.eval()
        with torch.no_grad(): mu = model(xt.to(dev)).cpu().numpy()
        e = per_step_error(mu, te["disp"])[:, -1].mean(); errs.append((n, e))
    cr = contraction_ratio(model, tens(gen_allo(n=20, n_eps=40, T=480, seed=9, **G))[0], dev)
    slope = (errs[-1][1] - errs[0][1])
    print(f"RESULT lam={a.lam} seed={a.seed} contraction_ratio={cr:.3f} err_vs_size={[(n, round(e,2)) for n,e in errs]} "
          f"grow(n6->n40)={slope:+.2f}")
