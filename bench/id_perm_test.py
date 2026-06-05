"""Test 1: id-permutation invariance. A purely RELATIONAL matcher uses only the equality structure of
landmark ids (which rays see the same id), not the absolute id value. So if we relabel the id alphabet
per episode at TEST time (a permutation that preserves equality structure), a relational model's error
must NOT change; a model that keys on absolute ids will change.

We test the GRU+landmark model (does it key on absolute ids?) with the no-landmark model as a control
(must be perfectly invariant — it never sees ids).
"""
import numpy as np, torch
from train_eval import cached, encode, train, AMB
from archs import build
from metrics import per_step_error

V = 12

def permute_ids(ds, seed=0):
    rng = np.random.default_rng(seed); lm = np.asarray(ds["ray_lm"]).copy(); out = lm.copy()
    for e in range(lm.shape[0]):
        perm = rng.permutation(V); mask = lm[e] >= 0; out[e][mask] = perm[lm[e][mask]]
    nds = dict(ds); nds["ray_lm"] = out; return nds

@torch.no_grad()
def final(model, ds, use_lm):
    x, _ = encode(ds, use_lm); return per_step_error(model(x).cpu().numpy(), ds["disp"])[:, -1].mean()

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    tr = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)
    xl, yl = encode(tr, True);  mlm = build("gru", xl.shape[2]); train(mlm, xl, yl, 60, "cpu")
    x0, y0 = encode(tr, False); m0  = build("gru", x0.shape[2]); train(m0,  x0, y0, 60, "cpu")
    print("\nid-permutation invariance (final error: normal -> ids-permuted; |delta| = reliance on ABSOLUTE id)")
    print(f"{'size':>6} | {'GRU+lm normal':>13} {'permuted':>9} {'|delta|':>8} | {'GRU-lm (control)':>16} {'permuted':>9}")
    for n, T in [(6,144),(12,288),(20,480)]:
        te = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
        tp = permute_ids(te, seed=n)
        a, b = final(mlm, te, True), final(mlm, tp, True)
        c, d = final(m0, te, False), final(m0, tp, False)
        print(f"{('n='+str(n)):>6} | {a:13.2f} {b:9.2f} {abs(b-a):8.2f} | {c:16.2f} {d:9.2f}")
    print("\n|delta|>0 for GRU+lm => it keys on ABSOLUTE ids (NOT purely relational). control should be ~0.")
