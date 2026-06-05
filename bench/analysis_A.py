"""Direction-A summary figure: the science story of TrackMaze, recomputed from scratch (no hardcoded
numbers). Four panels:
  (A) DRIFT LAW      — open-loop localization error vs step, at several maze sizes (lawful, size-scaling).
  (B) MEMORIZATION   — landmarks HURT: with-lm vs no-lm error across sizes for a GRU (benefit < 0).
  (C) DIVERSITY      — the harm shrinks as #distinct training worlds grows (shortcut suppressed by data).
  (D) ASSOCIATION    — oracle re-anchor degrades to useless as landmark AMBIGUITY rises (the threshold).
Saves figure_A.png + writes results.html (the demo page).
"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate import gen_dataset
from metrics import dead_reckon, per_step_error
from train_eval import cached, encode, train, AMB
from archs import build
from threshold_sweep import run_episode

def panelA(ax):
    for n, T, c in [(6,144,"#4c9be8"),(12,288,"#e8954c"),(20,480,"#d24c5a")]:
        ds = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
        est = np.stack([dead_reckon(ds["action"][e], (0.,0.), 0.) for e in range(ds["action"].shape[0])])
        curve = per_step_error(est, ds["disp"]).mean(0)
        ax.plot(np.linspace(0,1,len(curve)), curve, color=c, label=f"maze {2*n+1}x{2*n+1}")
    ax.set_title("(A) drift law: error grows lawfully, worse on bigger mazes")
    ax.set_xlabel("fraction of episode"); ax.set_ylabel("localization error (cells)"); ax.legend(fontsize=8)

def gru_final(use_lm, sizes, model):
    out = {}
    for n, T in sizes:
        te = cached(f"eval_n{n}_T{T}_a{AMB}", n=n, n_eps=60, T=T, ambiguity=AMB, seed=7)
        x,_ = encode(te, use_lm)
        with torch.no_grad(): pred = model(x).cpu().numpy()
        out[n] = per_step_error(pred, te["disp"])[:, -1].mean()
    return out

def panelB(ax):
    tr = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)
    sizes = [(6,144),(12,288),(20,480)]; res = {}
    for use_lm in (True, False):
        torch.manual_seed(0); x,y = encode(tr, use_lm); m = build("gru", x.shape[2]); train(m, x, y, 60, "cpu")
        res[use_lm] = gru_final(use_lm, sizes, m)
    ns = [n for n,_ in sizes]
    ax.plot(ns, [res[False][n] for n in ns], "-o", color="#3aa657", label="no landmarks (pure integration)")
    ax.plot(ns, [res[True][n] for n in ns], "-o", color="#d24c5a", label="+ landmarks")
    ax.set_title("(B) landmarks HURT: the net memorizes, doesn't correct")
    ax.set_xlabel("maze size n"); ax.set_ylabel("final error (cells)"); ax.legend(fontsize=8)

def panelC(ax):
    from diversity_sweep import gen_pool, final_err
    TOTAL, T = 128, 160; worlds = [4,16,64,256]; ben = []
    for nd in worlds:
        tr = gen_pool(n=6, n_distinct=nd, reps=max(1,TOTAL//nd), T=T, ambiguity=AMB, base_seed=1)
        r = {}
        for use_lm in (True, False):
            torch.manual_seed(0); x,y = encode(tr, use_lm); m = build("gru", x.shape[2]); train(m, x, y, 40, "cpu")
            te = cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)
            r[use_lm] = final_err(m, te, use_lm)
        ben.append(r[False] - r[True])
    ax.axhline(0, color="#888", lw=0.8)
    ax.plot(worlds, ben, "-o", color="#7a4ce8")
    ax.set_xscale("log", base=2); ax.set_title("(C) data diversity suppresses the shortcut")
    ax.set_xlabel("# distinct training mazes"); ax.set_ylabel("landmark benefit (>0 = helps)")

def panelD(ax):
    ambs = [0,1,2,3]; ol, rc = [], []
    for a in ambs:
        o, r = zip(*[run_episode(n=12, ambiguity=a, T=300, seed=1000+e) for e in range(30)])
        ol.append(np.mean(o)); rc.append(np.mean(r))
    ax.plot(ambs, rc, "-o", color="#e8954c", label="with oracle re-anchor")
    ax.plot(ambs, ol, "--", color="#888", label="open-loop (no cues)")
    ax.set_title("(D) cue ambiguity breaks correction (the threshold)")
    ax.set_xlabel("landmark ambiguity (0=unique → 3=very aliased)")
    ax.set_ylabel("steady-state error (cells)"); ax.set_xticks(ambs); ax.legend(fontsize=8)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    print("panel A (drift law)..."); panelA(axes[0,0])
    print("panel B (memorization)..."); panelB(axes[0,1])
    print("panel C (diversity)..."); panelC(axes[1,0])
    print("panel D (association threshold)..."); panelD(axes[1,1])
    fig.suptitle("TrackMaze — when does a net MEMORIZE vs TRACK, and how does spatial drift scale?", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig("figure_A.png", dpi=120)
    print("saved figure_A.png")
