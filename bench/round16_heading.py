"""Round 16: the now-ISOLATED core problem. The diagnostic showed: with oracle heading, even dumb
dead-reckon tracks a 49x49 traversal near-perfectly (GLOBAL 0.38); with drifting heading it's catastrophic
(31.6). So ALL of Stage-1 reduces to: can we estimate the agent's ABSOLUTE heading (incl. the 4-fold
quadrant) from observation? This trains a learned recurrent heading estimator and measures how well it does,
vs size, vs the freq-4 grid cue (mod-90 only) and command-integration (drifts).

The net gets per step: RAW omni view (geo+lm, at true heading -- NOT canonicalized) + last action. It can
integrate actions (-> command heading = an exact quadrant reference) AND read the grid (freq-4 -> heading
mod 90deg). If it FUSES them to recover absolute heading at low full-circle error across sizes -> heading is
solved -> plug into the tracker -> Stage-1 fixed. If it caps at the mod-90 ambiguity (errors clustering at
+-90/180) and grows with traversal length -> the 4-fold quadrant wall is confirmed.

Target: [cos(theta_rel), sin(theta_rel)], theta_rel = true heading - start heading.
"""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_head(n, n_eps, T, seed=0, loop=0.3):
    """Drive the DFS traversal; record RAW omni (geo,lm) + last-action one-hot + true heading-from-start."""
    rng = np.random.default_rng(seed); mv, rot = 0.22, 0.20
    GEO, LM, ACT, HEAD, GH, CMD = [], [], [], [], [], []
    for e in range(n_eps):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, seed=seed * 100000 + e, max_steps=10 ** 9, loop=loop)
        env.reset(); ang0 = env.ang; cmd = 0.0; last_a = 0
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
        g_, l_, a_, h_, gh_, c_ = [], [], [], [], [], []; done = False
        for (cx, cy) in walk[1:]:
            if done: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if len(g_) >= T: done = True; break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                err = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(err) > rot and err > 0) else (2 if abs(err) > rot else 0)
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                oa = np.zeros(4, np.float32); oa[last_a] = 1.0
                g_.append(g); l_.append(l); a_.append(oa)
                h_.append(wrap(env.ang - ang0))                 # TRUE heading-from-start (target)
                gh_.append(_grid_heading(g))                    # freq-4 grid cue (mod 90), baseline
                c_.append(cmd)                                  # command-integrated heading, baseline
                if a == 2: cmd -= rot
                elif a == 3: cmd += rot
                motion(env, a); last_a = a
        GEO.append(np.stack(g_)); LM.append(np.stack(l_)); ACT.append(np.stack(a_))
        HEAD.append(np.array(h_, np.float32)); GH.append(np.array(gh_, np.float32)); CMD.append(np.array(c_, np.float32))
    Tm = min(len(x) for x in GEO); cut = lambda L: np.stack([x[:Tm] for x in L])
    return dict(geo=cut(GEO), lm=cut(LM), act=cut(ACT), head=cut(HEAD), gh=cut(GH), cmd=cut(CMD))


class HeadNet(nn.Module):
    def __init__(self, din=68, h=128):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(din, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True); self.out = nn.Linear(h, 2)

    def forward(self, x):
        y, _ = self.gru(self.enc(x)); return F.normalize(self.out(y), dim=-1)


def feats(ds): return torch.tensor(np.concatenate([ds["geo"], ds["lm"], ds["act"]], -1), dtype=torch.float32)
def target(ds): return torch.tensor(np.stack([np.cos(ds["head"]), np.sin(ds["head"])], -1), dtype=torch.float32)


def ang_err_deg(pred_cs, true_h):
    pa = np.arctan2(pred_cs[..., 1], pred_cs[..., 0])
    return np.abs((pa - true_h + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr = [gen_head(n, 200, 40 * n, seed=1) for n in a.train_sizes]
    ev = {n: gen_head(n, 30, 40 * n, seed=7) for n in a.eval_sizes}
    Xtr = [feats(t).to(DEV) for t in tr]; Ytr = [target(t).to(DEV) for t in tr]
    model = HeadNet().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3); rng = np.random.default_rng(a.seed)
    for it in range(a.epochs):
        k = rng.integers(len(tr)); idx = rng.integers(0, Xtr[k].shape[0], 64)
        pred = model(Xtr[k][idx]); loss = ((pred - Ytr[k][idx]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Round 16 heading: mean full-circle heading error (deg) vs size. learned vs grid(freq-4) vs cmd.")
    print(f"{'n':>4} {'grid':>8} | {'LEARNED':>8} {'grid(mod90)':>11} {'cmd':>7}")
    for n in a.eval_sizes:
        ds = ev[n]; X = feats(ds)
        with torch.no_grad():
            preds = np.concatenate([model(X[i:i+32].to(DEV)).cpu().numpy() for i in range(0, X.shape[0], 32)])
        le = ang_err_deg(preds, ds["head"]).mean()
        ge = ang_err_deg(np.stack([np.cos(ds["gh"]), np.sin(ds["gh"])], -1), ds["head"]).mean()  # raw grid (mod90, ambiguous)
        ce = ang_err_deg(np.stack([np.cos(ds["cmd"]), np.sin(ds["cmd"])], -1), ds["head"]).mean()
        tag = "*tr*" if n in a.train_sizes else ""
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8}{tag:>4} | {le:8.1f} {ge:11.1f} {ce:7.1f}")
        if a.out:
            with open(a.out, "a") as f:
                f.write(f"learned {a.seed} {n} {le:.2f}\ngrid {a.seed} {n} {ge:.2f}\ncmd {a.seed} {n} {ce:.2f}\n")
    print("\nLEARNED << 90 and flat with size => quadrant resolved, heading SOLVED. ~mod90-capped & growing => 4-fold wall.")
