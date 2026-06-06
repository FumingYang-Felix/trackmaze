"""Train + eval the AlloTracker (Stage 1) with the allo-canonical place code.
Two-stage: (A) contrastive on the place code (same true cell -> similar), (B) integrator + EMA correction
with GT-bound confidence. Reports ON(correction) vs OFF(integrator) final error per OOD size -- the
question: does the allo descriptor finally make the correction HELP (ON < OFF)?

  python train_allo.py --canon true   # oracle heading (does a perfect allo code work)
  python train_allo.py --canon cmd    # command-integrated heading (realistic)
"""
import argparse, numpy as np, torch, torch.nn as nn
from generate_allo import gen_allo
from metrics import per_step_error
from arch_allo import AlloTracker

def tens(ds):
    geo = torch.from_numpy(ds["canon_geo"]); lm = torch.from_numpy(ds["canon_lm"])
    act = torch.from_numpy(ds["action"]); oh = torch.zeros(*act.shape, 4); oh.scatter_(2, act.unsqueeze(-1), 1.0)
    return geo, lm, oh, torch.from_numpy(ds["disp"]), torch.from_numpy(ds["cell"])

def contrastive(v, cells, gap=8, temp=0.07):
    B, T, _ = v.shape; tot = 0.0; nb = 0
    ti = torch.arange(T, device=v.device); far = (ti[:, None] - ti[None, :]).abs() > gap
    for b in range(B):
        sim = (v[b] @ v[b].t()) / temp; same = cells[b][:, None] == cells[b][None, :]
        pos = same & far; anc = pos.any(1)
        if anc.sum() == 0: continue
        sim = sim.masked_fill(~far, -1e9); lse = torch.logsumexp(sim, 1)
        ps = (torch.exp(sim) * pos).sum(1); tot = tot - (torch.log(ps + 1e-9) - lse)[anc].mean(); nb += 1
    return tot / max(nb, 1)

def revisit_consistency(mu, cells, gap=8):
    """Force 'same true cell -> same latent position': penalize the distance between estimates at
    different times that share a true cell (revisits). The self-correcting / loop-closure-for-free
    objective -- pressure to USE observations to keep the estimate consistent, not a bolt-on snap."""
    B, T, _ = mu.shape; tot = 0.0; nb = 0
    ti = torch.arange(T, device=mu.device); far = (ti[:, None] - ti[None, :]).abs() > gap
    for b in range(B):
        same = (cells[b][:, None] == cells[b][None, :]) & far
        if same.sum() == 0: continue
        d = torch.cdist(mu[b], mu[b]); tot = tot + d[same].mean(); nb += 1
    return tot / max(nb, 1)

def train(model, geo, lm, oh, y, cells, dev, ea, eb, bs=32, lr=1e-3, beta=1.0, consistency=0.0):
    model.to(dev); geo, lm, oh, y, cells = [t.to(dev) for t in (geo, lm, oh, y, cells)]; E = geo.shape[0]
    vp = list(model.view.parameters()); optA = torch.optim.Adam(vp, lr)
    for ep in range(ea):                                   # Stage A: place code
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; optA.zero_grad()
            contrastive(model.encode_view(geo[b], lm[b]), cells[b]).backward(); optA.step()
    for p in vp: p.requires_grad = False
    rest = [p for p in model.parameters() if p.requires_grad]
    optB = torch.optim.Adam(rest, lr); mse = nn.MSELoss(); bce = nn.BCELoss()
    for ep in range(eb):                                   # Stage B: integrator + correction
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; optB.zero_grad()
            mu, v, mu_prior, g, lbl = model(geo[b], lm[b], oh[b], gt_disp=y[b], return_aux=True)
            loss = mse(mu, y[b]) + mse(mu_prior, y[b])
            if g is not None: loss = loss + beta * bce(g, lbl)
            if consistency > 0: loss = loss + consistency * revisit_consistency(mu, cells[b])
            loss.backward(); optB.step()
    return model

@torch.no_grad()
def final_err(model, ds, dev, use_memory):
    model.eval(); model.use_memory = use_memory; geo, lm, oh, y, _ = tens(ds)
    mu = model(geo.to(dev), lm.to(dev), oh.to(dev)).cpu().numpy()
    return per_step_error(mu, ds["disp"])[:, -1].mean()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default="cmd"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--device", default="cpu")
    ap.add_argument("--loop", type=float, default=0.0); ap.add_argument("--rot_noise", type=float, default=0.09)
    ap.add_argument("--tscale", type=float, default=1.0); ap.add_argument("--train_n", type=int, default=6)
    ap.add_argument("--ema_gain", type=float, default=0.3); ap.add_argument("--store_gap", type=int, default=3)
    ap.add_argument("--mask_recent", type=int, default=12); ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--consistency", type=float, default=0.0)   # wave-2: revisit-consistency loss weight
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = a.device
    G = dict(ambiguity=1, canon=a.canon, rot_noise=a.rot_noise, loop=a.loop)
    tr = gen_allo(n=a.train_n, n_eps=200, T=int(160*a.tscale), seed=1, **G)
    geo, lm, oh, y, cells = tens(tr)
    model = AlloTracker(store_gap=a.store_gap, mask_recent=a.mask_recent, ema_gain=a.ema_gain, topk=a.topk)
    train(model, geo, lm, oh, y, cells, dev, a.epochs, a.epochs, consistency=a.consistency)
    bs, large = [], []
    for n in (6, 12, 20, 28):
        te = gen_allo(n=n, n_eps=60, T=int(24*n*a.tscale), seed=7, **G)
        off = final_err(model, te, dev, False); on = final_err(model, te, dev, True)
        bs.append(off - on)
        if n >= 20: large.append(off - on)
    cfg = (f"canon={a.canon} loop={a.loop} rot={a.rot_noise} tscale={a.tscale} trainN={a.train_n} "
           f"ema={a.ema_gain} gap={a.store_gap} recent={a.mask_recent} topk={a.topk} cons={a.consistency} seed={a.seed}")
    print(f"RESULT [{cfg}] LARGE_benefit={np.mean(large):+.3f} mean={np.mean(bs):+.3f} "
          f"per_size={[round(b,2) for b in bs]}")
