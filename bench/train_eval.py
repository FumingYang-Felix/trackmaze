"""Unified train+eval for one (architecture, seed). Train on the SMALL maze, evaluate on the train size
and on progressively LARGER (OOD) sizes -- with the episode length scaled to maze size so the size axis
actually bites (drift has room to accumulate). Writes a results JSON.

All runs share the SAME cached train/eval data, so differences are the architecture (+init seed), not the
data -- the seed-variance lesson from the Memory Maze probe. Reference lines: open-loop dead-reckoner
(floor of "use no landmarks") and we already know the oracle-association ceiling ~0.4 cells.

  python train_eval.py --arch transformer --seed 0
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from generate import gen_dataset
from metrics import per_step_error, drift_curve, correction_curve, dead_reckon
from archs import build

V = 12                       # landmark vocab (ambiguity=1 -> ids transfer across sizes)
AMB = 1
TRAIN_N, TRAIN_T, TRAIN_EPS = 6, 160, 200
EVAL_SIZES = [(6,144),(9,216),(12,288),(16,384)]   # (n, T): T ~ 24*n so bigger maze = longer horizon
EVAL_EPS = 60
CACHE = os.path.join(os.path.dirname(__file__), "_cache"); os.makedirs(CACHE, exist_ok=True)

def cached(tag, **kw):
    p = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
    ds = gen_dataset(**kw); np.savez(p, **{k: v for k, v in ds.items()}); return ds

def encode(ds, use_landmarks=True):
    feat = torch.from_numpy(np.asarray(ds["feat"], dtype=np.float32))
    lm = np.asarray(ds["ray_lm"]); E,T,K = lm.shape
    oh = np.zeros((E,T,K,V+1), dtype=np.float32)
    if use_landmarks:
        idx = np.clip(lm+1, 0, V).astype(np.int64); np.put_along_axis(oh, idx[...,None], 1.0, axis=3)
    x = torch.cat([feat, torch.from_numpy(oh.reshape(E,T,K*(V+1)))], dim=2)
    return x, torch.from_numpy(np.asarray(ds["disp"], dtype=np.float32))

def train(model, x, y, epochs, dev, bs=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); lf = nn.MSELoss(); model.to(dev); x=x.to(dev); y=y.to(dev)
    E = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(E)
        for i in range(0, E, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lf(model(x[b]), y[b]).backward(); opt.step()
    return model

@torch.no_grad()
def evaluate(model, ds, dev, use_lm=True):
    x,_ = encode(ds, use_lm); model.eval(); pred = model(x.to(dev)).cpu().numpy()
    errs = per_step_error(pred, ds["disp"])
    ol = np.stack([dead_reckon(ds["action"][e], (0.,0.), 0.) for e in range(ds["action"].shape[0])])
    oe = per_step_error(ol, ds["disp"])
    cc = {g: v for g, v, _ in correction_curve(errs, ds["reanchor"], 30)}
    return dict(final=float(errs[:,-1].mean()), open_final=float(oe[:,-1].mean()),
                drift=drift_curve(errs, 8), corr={k: round(v,3) for k,v in cc.items()})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_landmarks", action="store_true")   # ablate landmark input -> pure path integration
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device; use_lm = not a.no_landmarks

    tr = cached(f"train_n{TRAIN_N}_T{TRAIN_T}_a{AMB}", n=TRAIN_N, n_eps=TRAIN_EPS, T=TRAIN_T, ambiguity=AMB, seed=1)
    x, y = encode(tr, use_lm); din = x.shape[2]
    model = build(a.arch, din); train(model, x, y, a.epochs, dev)

    res = {"arch": a.arch, "seed": a.seed, "din": din, "landmarks": use_lm, "by_size": {}}
    for n, T in EVAL_SIZES:
        te = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=EVAL_EPS, T=T, ambiguity=AMB, seed=7)
        res["by_size"][n] = evaluate(model, te, dev, use_lm)
    tag = "" if use_lm else "_nolm"
    out = os.path.join(CACHE, f"res_{a.arch}_s{a.seed}{tag}.json")
    with open(out, "w") as f: json.dump(res, f, indent=2)
    print(f"[{a.arch} seed{a.seed}{tag}] " + "  ".join(
        f"n{n}:{res['by_size'][n]['final']:.2f}(ol {res['by_size'][n]['open_final']:.2f})" for n,_ in EVAL_SIZES))

if __name__ == "__main__":
    main()
