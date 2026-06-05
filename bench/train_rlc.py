"""Train + evaluate the Relational Loop-Closure tracker (arch_rlc).

Loss = position MSE (on the memory-corrected estimate) + lambda * supervised CONTRASTIVE loss on the view
embeddings (same true cell, temporally distant => pull together; different cell => push apart). The
contrastive term is the signal that makes the matching RELATIONAL and transferable -- without it the model
has no reason to learn "same place?" and falls back to memorizing.

Headline ablation: evaluate the SAME trained model with memory ON vs OFF. ON<OFF => the loop-closure
correction actually helps (positive landmark benefit -- the thing every baseline failed to get).

  python train_rlc.py --seed 0
References: oracle ~0.40 (ceiling) | open-loop 0.76-1.26 | integrator floor ~1.1-1.9 | baselines+lm 1.6-2.9
"""
import argparse, numpy as np, torch, torch.nn as nn
from train_eval import cached, encode, EVAL_SIZES, EVAL_EPS, TRAIN_N, TRAIN_T, TRAIN_EPS, AMB
from metrics import per_step_error, drift_curve, dead_reckon
from arch_rlc import RLCTracker

def cell_ids(pos):
    c = np.floor(np.asarray(pos)).astype(np.int64)          # (E,T,2) -> integer cell
    return torch.from_numpy(c[..., 1] * 1000 + c[..., 0])   # (E,T) hashed cell id

def contrastive(v, cells, gap=8, temp=0.07):
    """within-episode supervised contrastive over temporally-distant pairs (the loop-closure signal)."""
    B, T, _ = v.shape; tot = 0.0; n = 0
    ti = torch.arange(T, device=v.device); far = (ti[:, None] - ti[None, :]).abs() > gap
    for b in range(B):
        sim = (v[b] @ v[b].t()) / temp
        same = cells[b][:, None] == cells[b][None, :]
        pos = same & far
        anc = pos.any(1)
        if anc.sum() == 0: continue
        sim = sim.masked_fill(~far, -1e9)
        lse = torch.logsumexp(sim, dim=1)
        pos_sum = (torch.exp(sim) * pos).sum(1)
        l = -(torch.log(pos_sum + 1e-9) - lse)
        tot = tot + l[anc].mean(); n += 1
    return tot / max(n, 1)

def train(model, x, y, cells, epochs, dev, bs=32, lr=1e-3, lam=0.5, beta=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr); mse = nn.MSELoss(); bce = nn.BCELoss()
    model.to(dev); x, y, cells = x.to(dev), y.to(dev), cells.to(dev); E = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E); tot = tc = tcf = 0.0
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            mu, v, mu_prior, g, lbl = model(x[b], gt_disp=y[b], return_aux=True)
            lm = mse(mu, y[b]) + 1.0 * mse(mu_prior, y[b])      # protect the landmark-free integrator floor
            lc = contrastive(v, cells[b])
            lcf = bce(g.clamp(1e-6, 1-1e-6), lbl) if g is not None else torch.zeros((), device=dev)  # GT-bound confidence
            (lm + lam * lc + beta * lcf).backward(); opt.step(); tot += lm.item(); tc += lc.item(); tcf += float(lcf)
        if (ep+1) % 15 == 0:
            nb = (E+bs-1)//bs; print(f"    epoch {ep+1:3d}  pos {tot/nb:.3f}  contrast {tc/nb:.3f}  conf-bce {tcf/nb:.3f}")
    return model

@torch.no_grad()
def final_err(model, ds, dev, use_memory):
    model.eval(); model.use_memory = use_memory
    x,_ = encode(ds); mu = model(x.to(dev)).cpu().numpy()
    return per_step_error(mu, ds["disp"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--device", default="cpu")
    ap.add_argument("--lam", type=float, default=0.5); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = a.device

    tr = cached(f"train_n{TRAIN_N}_T{TRAIN_T}_a{AMB}", n=TRAIN_N, n_eps=TRAIN_EPS, T=TRAIN_T, ambiguity=AMB, seed=1)
    x, y = encode(tr); cells = cell_ids(tr["pos"]); din = x.shape[2]
    model = RLCTracker(din); train(model, x, y, cells, a.epochs, dev, lam=a.lam)

    print(f"\n[RLC v2 seed{a.seed}] final error (cells). OFF=integrator only, ON=GT-bound calibrated correction")
    print(f"{'size':>6}  {'OFF(int)':>9}  {'ON(corr)':>9}  {'benefit':>9}  {'open':>6}")
    for n, T in EVAL_SIZES:
        te = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=EVAL_EPS, T=T, ambiguity=AMB, seed=7)
        off = final_err(model, te, dev, False)[:, -1].mean()
        on  = final_err(model, te, dev, True)[:, -1].mean()
        ol = per_step_error(np.stack([dead_reckon(te["action"][e], (0.,0.), 0.) for e in range(te["action"].shape[0])]), te["disp"])[:, -1].mean()
        print(f"{('n='+str(n)):>6}  {off:9.2f}  {on:9.2f}  {off-on:+9.2f}  {ol:6.2f}")

    # --- confidence calibration on an UNSEEN maze (n=12): does the trust head fire ONLY on helpful matches? ---
    te = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=EVAL_EPS, T=288, ambiguity=AMB, seed=7)
    model.use_memory = True; x, yv = encode(te)
    with torch.no_grad():
        _, v, _, g, lbl = model(x.to(dev), gt_disp=yv.to(dev), return_aux=True)
    g, lbl = g.cpu().reshape(-1), lbl.cpu().reshape(-1); fire = g > 0.5
    prec = (lbl[fire].mean().item() if fire.any() else float('nan'))
    rec = ((g[lbl > 0.5] > 0.5).float().mean().item() if (lbl > 0.5).any() else float('nan'))
    print(f"\nconfidence (n=12 unseen): fires {fire.float().mean().item():.2f} of steps | when it fires it's RIGHT "
          f"{prec:.2f} (precision) | catches {rec:.2f} of true matches (recall)")

if __name__ == "__main__":
    main()
