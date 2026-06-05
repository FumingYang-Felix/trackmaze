"""Core's experiment: is the landmark-memorization shortcut suppressed by DATA DIVERSITY?

Hypothesis: with few distinct training worlds the recurrent net memorizes each (per-world lookup) ->
landmarks HURT on held-out mazes; as the number of distinct worlds grows past the net's memorization
capacity the shortcut dies -> the landmark benefit (no-lm err minus with-lm err) climbs toward 0/positive.

We hold the TOTAL training data fixed (same #episodes, #epochs) and vary only how many DISTINCT mazes
those episodes are drawn from (fewer worlds => more repeats each). Same GRU, trained with and without the
landmark input, evaluated on held-out mazes (the standard eval cache). The benefit-vs-diversity curve is
the answer.
"""
import numpy as np, torch
from env import TrackMazeEnv
from generate import explore_policy, egocentric_disp
from train_eval import encode, train, cached, AMB
from archs import build
from metrics import per_step_error

def gen_pool(n, n_distinct, reps, T, ambiguity, base_seed):
    rng = np.random.default_rng(base_seed)
    FEAT, LM, POS, DISP = [], [], [], []
    for m in range(n_distinct):
        env = TrackMazeEnv(n=n, ambiguity=ambiguity, seed=base_seed*100000 + m, max_steps=T+2)
        for r in range(reps):
            obs = env.reset()
            f, lm, po = [], [], [np.array([env.px, env.py], np.float32)]
            ang0 = env.ang
            for t in range(T):
                a = explore_policy(obs, rng)
                f.append(np.concatenate([obs["rays"], obs["last_action"]]).astype(np.float32))
                lm.append(obs["ray_lm"].astype(np.int32))
                obs, _, _, gt = env.step(a); po.append(gt["pos"])
            pos = np.stack(po[:T])
            FEAT.append(np.stack(f)); LM.append(np.stack(lm)); POS.append(pos); DISP.append(egocentric_disp(pos, ang0))
    return dict(feat=np.stack(FEAT), ray_lm=np.stack(LM), pos=np.stack(POS), disp=np.stack(DISP))

def final_err(model, ds, use_lm, dev="cpu"):
    x, _ = encode(ds, use_lm)
    with torch.no_grad(): pred = model(x.to(dev)).cpu().numpy()
    return per_step_error(pred, ds["disp"])[:, -1].mean()

if __name__ == "__main__":
    TOTAL, T = 256, 160
    EVALS = [(6, 144), (12, 288)]
    print("diversity sweep: total episodes fixed = 256, vary #distinct mazes; GRU +lm vs -lm on held-out")
    print(f"{'#worlds':>8} {'repeats':>8} | " + " ".join(f"n{n}:benefit(+lm vs -lm)" for n, _ in EVALS))
    for nd in [4, 16, 64, 256]:
        reps = max(1, TOTAL // nd)
        tr = gen_pool(n=6, n_distinct=nd, reps=reps, T=T, ambiguity=AMB, base_seed=1)
        res = {}
        for use_lm in (True, False):
            torch.manual_seed(0); np.random.seed(0)
            x, y = encode(tr, use_lm); m = build("gru", x.shape[2]); train(m, x, y, 60, "cpu")
            res[use_lm] = {n: final_err(m, cached(f"eval_n{n}_T{T2}_a{AMB}", n=n, n_eps=60, T=T2, ambiguity=AMB, seed=7), use_lm)
                           for n, T2 in EVALS}
        ben = " ".join(f"n{n}:{res[False][n]-res[True][n]:+.2f} (+lm {res[True][n]:.2f} / -lm {res[False][n]:.2f})" for n, _ in EVALS)
        print(f"{nd:>8} {reps:>8} | {ben}")
    print("\n>0 benefit = landmarks HELP (shortcut suppressed); <0 = landmarks HURT (memorized).")
