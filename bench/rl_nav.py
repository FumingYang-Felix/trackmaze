"""Round 13 (option 1): a fully-learned navigation agent (NO oracle control). Recurrent GRU policy trained
by PPO on SMALL mazes to reach the exit with FEWER steps (reward = goal bonus - per-step cost), evaluated
OOD on larger mazes. The agent outputs raw actions (forward/back/turn) from egocentric observations only --
this is the real Stage-2 the earlier nav tests bypassed with oracle low-level execution. Tests whether a
learned policy generalizes the navigation strategy in size (reach-rate + steps vs size), vs the phase-1
finding that standard nets don't transfer.

Observation per step: omni 32-ray geometry + landmark-presence (rolled to the command frame) + last action
one-hot + a bias. Reward: +R_goal on reaching the exit, -step_cost each step, small shaping toward the goal
is OFF by default (pure sparse+step-cost to test genuine exploration). Recurrent PPO (GAE), full-episode
sequences. Periodic OOD eval (reach-rate, mean steps) at sizes unseen in training.
"""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NACT = 4  # 0 fwd, 1 back, 2 turn-left, 3 turn-right


def geo_dist(env, n):
    """BFS geodesic distance (in cells) from every cell to the goal cell -> potential for reward shaping."""
    adj = cell_graph(env.wall, n)
    goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    dist = {goal: 0}; q = deque([goal])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist: dist[v] = dist[u] + 1; q.append(v)
    return dist


def obs_vec(env, cmd, last_a):
    g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
    sh = int(round(cmd / (2 * math.pi) * KO))
    g = np.roll(g, sh); l = np.roll(l, sh)
    a = np.zeros(NACT, np.float32); a[last_a] = 1.0
    return np.concatenate([g, l, a, [1.0]]).astype(np.float32)   # (32+32+4+1 = 69)


class Policy(nn.Module):
    def __init__(self, din=69, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True)
        self.pi = nn.Linear(h, NACT); self.v = nn.Linear(h, 1)

    def forward(self, x, hcell=None):
        z = self.enc(x)
        y, hn = self.gru(z, hcell)
        return self.pi(y), self.v(y).squeeze(-1), hn

    def step(self, x1, hcell):  # x1: (B,din) single step
        z = self.enc(x1).unsqueeze(1)
        y, hn = self.gru(z, hcell)
        return self.pi(y[:, 0]), self.v(y[:, 0]).squeeze(-1), hn


def rollout_episode(env, model, n, max_steps, step_cost, r_goal, greedy=False, shape=0.0):
    """Run one episode with the policy controlling actions. Returns trajectory tensors + (reached, steps).
    shape>0 adds potential-based geodesic shaping (training signal; theory-preserving, uses true goal dist)."""
    env.reset(); cmd = 0.0; last_a = 0
    h = torch.zeros(1, 1, model.gru.hidden_size, device=DEV)
    dist = geo_dist(env, n) if shape > 0 else None
    def cell(): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)
    prev_phi = (-dist.get(cell(), 2 * n)) if shape > 0 else 0.0
    obss, acts, logps, vals, rews = [], [], [], [], []
    reached = False
    for t in range(max_steps):
        o = obs_vec(env, cmd, last_a)
        ot = torch.tensor(o, device=DEV).unsqueeze(0)
        with torch.no_grad():
            logits, val, h = model.step(ot, h)
            probs = F.softmax(logits, -1)
            a = int(torch.argmax(probs, -1)) if greedy else int(torch.multinomial(probs, 1))
            logp = float(torch.log_softmax(logits, -1)[0, a])
        obss.append(o); acts.append(a); logps.append(logp); vals.append(float(val))
        if a == 2: cmd -= 0.20
        elif a == 3: cmd += 0.20
        _, _, done, gt = env.step(a)
        last_a = a
        rr = -step_cost
        if shape > 0:
            phi = -dist.get(cell(), 2 * n)
            rr += shape * (0.99 * phi - prev_phi); prev_phi = phi   # potential-based shaping
        if math.hypot(env.gx - env.px, env.gy - env.py) < 0.6:
            rr += r_goal; reached = True
        rews.append(rr)
        if reached or done: break
    return dict(obs=np.array(obss, np.float32), act=np.array(acts), logp=np.array(logps, np.float32),
                val=np.array(vals, np.float32), rew=np.array(rews, np.float32)), reached, len(acts)


def gae(rew, val, gamma=0.99, lam=0.95):
    T = len(rew); adv = np.zeros(T, np.float32); last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + val


def evaluate(model, sizes, mazes, max_mult, step_cost, r_goal, seed0=40000):
    res = {}
    for n in sizes:
        rr, st = [], []
        for m in range(mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
            _, reached, steps = rollout_episode(env, model, n, max_mult * n * n, step_cost, r_goal, greedy=True)
            rr.append(1 if reached else 0); st.append(steps if reached else np.nan)
        res[n] = (float(np.mean(rr)), float(np.nanmean(st)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--eps_per_iter", type=int, default=32)
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[5, 7]); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--step_cost", type=float, default=0.01); ap.add_argument("--r_goal", type=float, default=5.0)
    ap.add_argument("--max_mult", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28])
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--shape", type=float, default=0.5); ap.add_argument("--curriculum", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = Policy().to(DEV); opt = torch.optim.Adam(model.parameters(), a.lr)
    rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"device={DEV} train_sizes={a.train_sizes}", flush=True)
    for it in range(a.iters):
        # collect episodes
        traj = []; reach = []; steps = []
        # curriculum: ramp the max train size from small to full over the first 60% of training
        if a.curriculum:
            frac = min(1.0, it / (0.6 * a.iters)); hi = 1 + int(frac * (max(a.train_sizes) - min(a.train_sizes)))
            sizes_now = [s for s in a.train_sizes if s <= min(a.train_sizes) + hi] or [min(a.train_sizes)]
        else:
            sizes_now = a.train_sizes
        for e in range(a.eps_per_iter):
            n = sizes_now[rng.integers(len(sizes_now))]
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9)
            tr, rd, stp = rollout_episode(env, model, n, a.max_mult * n * n, a.step_cost, a.r_goal, shape=a.shape)
            adv, ret = gae(tr["rew"], tr["val"]); tr["adv"] = adv; tr["ret"] = ret; traj.append(tr)
            reach.append(rd); steps.append(stp)
        # PPO update (per-episode sequences; flatten for the loss)
        allo = np.concatenate([t["obs"] for t in traj]); alla = np.concatenate([t["act"] for t in traj])
        alllp = np.concatenate([t["logp"] for t in traj]); alladv = np.concatenate([t["adv"] for t in traj])
        allret = np.concatenate([t["ret"] for t in traj])
        adv_n = (alladv - alladv.mean()) / (alladv.std() + 1e-6)
        O = torch.tensor(allo, device=DEV); A = torch.tensor(alla, device=DEV)
        LP = torch.tensor(alllp, device=DEV); AD = torch.tensor(adv_n, device=DEV); RET = torch.tensor(allret, device=DEV)
        for _ in range(a.epochs):
            logits, val, _ = model(O.unsqueeze(0))
            logits = logits[0]; val = val[0]
            lp = torch.log_softmax(logits, -1).gather(1, A.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(lp - LP)
            l_pi = -torch.min(ratio * AD, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * AD).mean()
            l_v = F.mse_loss(val, RET)
            ent = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean()
            loss = l_pi + 0.5 * l_v - 0.01 * ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (it + 1) % max(1, a.iters // 20) == 0:
            print(f"it {it+1}/{a.iters} reach={np.mean(reach):.2f} steps={np.mean(steps):.0f} "
                  f"loss={loss.item():.2f} ent={ent.item():.2f} ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== OOD eval (greedy): reach-rate / mean-steps vs size ===", flush=True)
    res = evaluate(model, a.eval_sizes, 20, a.max_mult, a.step_cost, a.r_goal)
    for n in a.eval_sizes:
        tag = "*tr*" if n in a.train_sizes else ""
        print(f"  n={n:>2} {f'{2*n+1}x{2*n+1}':>8}{tag:>4}: reach={res[n][0]:.2f} steps={res[n][1]:.0f}", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for n in a.eval_sizes:
                f.write(f"{a.tag} {a.seed} {n} {res[n][0]:.3f} {res[n][1]:.1f}\n")
