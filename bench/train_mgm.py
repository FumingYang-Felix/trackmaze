"""Round 8 training: train MGM and a plain-GRU baseline on SMALL mazes (n=6,12), evaluate OOD on large
(n=20,28,40). Predict egocentric displacement-from-start. Report LOCAL error (relative displacement over a
window = navigable accuracy) and GLOBAL error (final displacement). Thesis: the motion-gated-memory arch
keeps LOCAL flat and GLOBAL growing SLOWER than the plain GRU across sizes (the GRU memorizes a landmark
vocabulary that doesn't transfer -- phase-1)."""
import argparse, time, os, numpy as np, torch, torch.nn as nn
from generate_allo import gen_allo
from arch_mgm import build

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tens(ds):
    g = torch.tensor(ds["canon_geo"], dtype=torch.float32)
    l = torch.tensor(ds["canon_lm"], dtype=torch.float32)
    m = torch.tensor(ds["cmd_step"], dtype=torch.float32)
    y = torch.tensor(ds["disp"], dtype=torch.float32)
    return g, l, m, y


def gerr(p, t): return np.linalg.norm(p - t, axis=2)[:, -1].mean()
def lerr(p, t, W=30):
    rp = p[:, W:] - p[:, :-W]; rt = t[:, W:] - t[:, :-W]
    return np.linalg.norm(rp - rt, axis=2).mean()


def train(arch, train_sets, epochs, bs, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(arch).to(DEV); opt = torch.optim.Adam(model.parameters(), lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        # sample a train size, then a minibatch of episodes from it
        g, l, m, y = train_sets[rng.integers(len(train_sets))]
        idx = rng.integers(0, g.shape[0], bs)
        gb, lb, mb, yb = g[idx].to(DEV), l[idx].to(DEV), m[idx].to(DEV), y[idx].to(DEV)
        pred = model(gb, lb, mb)
        loss = ((pred - yb) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  [{arch}] ep {ep+1}/{epochs} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, ds):
    g, l, m, y = tens(ds)
    preds = []
    for i in range(0, g.shape[0], 64):
        preds.append(model(g[i:i+64].to(DEV), l[i:i+64].to(DEV), m[i:i+64].to(DEV)).cpu().numpy())
    pred = np.concatenate(preds)
    return lerr(pred, ds["disp"]), gerr(pred, ds["disp"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archs", nargs="+", default=["mgm", "gru"])
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--out", default="")   # if set, append "arch seed n local global" lines for aggregation
    ap.add_argument("--save", default="")   # if set, torch.save the trained model state_dict here
    a = ap.parse_args()
    print(f"device={DEV} train_sizes={a.train_sizes} eval_sizes={a.eval_sizes} epochs={a.epochs}", flush=True)
    G = dict(ambiguity=1, canon="cmd", rot_noise=0.09, loop=0.3)
    t0 = time.time()
    train_sets = [tens(gen_allo(n=n, n_eps=300, T=24 * n, seed=1, **G)) for n in a.train_sizes]
    eval_sets = {n: gen_allo(n=n, n_eps=60, T=24 * n, seed=7, **G) for n in a.eval_sizes}
    print(f"data ready ({time.time()-t0:.0f}s)", flush=True)
    results = {}
    for arch in a.archs:
        model = train(arch, train_sets, a.epochs, a.bs, a.lr, a.seed)
        if a.save:
            torch.save(model.state_dict(), a.save); print(f"saved {arch} -> {a.save}", flush=True)
        results[arch] = {n: evaluate(model, eval_sets[n]) for n in a.eval_sizes}
        if a.out:
            with open(a.out, "a") as f:
                for n in a.eval_sizes:
                    lo, gl = results[arch][n]
                    f.write(f"{arch} {a.seed} {n} {lo:.4f} {gl:.4f}\n")
    print(f"\n{'n':>4} {'grid':>8} | " + " | ".join(f"{arch+' LOCAL/GLOBAL':>18}" for arch in a.archs))
    for n in a.eval_sizes:
        tag = "*tr*" if n in a.train_sizes else ""
        seg = " | ".join(f"{results[arch][n][0]:7.2f} {results[arch][n][1]:7.2f}" for arch in a.archs)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8}{tag:>4} | {seg}")
    print("\nGROWTH (LOCAL, GLOBAL) across eval sizes:")
    for arch in a.archs:
        L = [results[arch][n][0] for n in a.eval_sizes]; Gl = [results[arch][n][1] for n in a.eval_sizes]
        print(f"  {arch:>4}: LOCAL {L[0]:.2f}->{L[-1]:.2f} ({L[-1]-L[0]:+.2f})  GLOBAL {Gl[0]:.2f}->{Gl[-1]:.2f} ({Gl[-1]-Gl[0]:+.2f})")
