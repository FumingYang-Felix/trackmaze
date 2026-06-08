"""Round 13 (the REAL Stage-2): a fully-learned navigation policy that ACTUALLY WALKS the maze (no oracle
low-level execution), trained by PPO on SMALL mazes, evaluated OOD on LARGER mazes. Tests the spotlight claim
for navigation: a learned policy generalizes the navigation strategy in SIZE iff it has the right inductive
bias (R12: drift-free topological/relational state from heading-mod-90 LABELS + a spatial memory that grows
with area), where end-to-end policies (raw GRU / Transformer) DRIFT and collapse OOD.

This file fixes two things vs the earlier rl_nav.py sketch:
  (1) CORRECT recurrent PPO: episodes are padded into a batch and the GRU is run per-episode (h0=0 each), so the
      recomputed log-probs match the behavior log-probs (rl_nav.py ran one GRU over the flat concatenation of all
      episodes -> hidden state bled across boundaries -> broken ratio).
  (2) An --arch switch: gru_raw | gru_grid | transformer | ours, on the SAME task/reward, to isolate the
      inductive bias as the cause of OOD-size transfer.

Obs per step (egocentric, partial): omni 32-ray geometry + 32 landmark-presence, rolled to the COMMAND frame
(the agent's integrated commanded heading -- drift-free vs its own commands), + last action one-hot + bias.
'ours' additionally reads a small neighborhood of a growing SPATIAL MEMORY addressed by DRIFT-FREE topological
coords (built from the grid-mod-90 heading label, not the drifting command/metric estimate).
Reward: +novelty for a new cell (revisits neutral -> backtrack allowed, no double penalty), -step_cost, +r_goal."""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NACT = 4  # 0 fwd, 1 back, 2 turn-left, 3 turn-right
GBASE = 32 + 32 + NACT + 1   # omni geom + lm presence + last-action + bias = 69


# ---------------------------------------------------------------- grid-mod-90 heading (allothetic, drift-free)
def grid_phase(g):
    """freq-4 Fourier phase of the omni geometry ray vector -> heading mod 90deg (the allothetic wall cue).
    Returns (cos2,sin2) of the doubled-phase so it is a smooth 2D feature; and the integer 'fine bin' is the
    drift-free orientation reference used by 'ours' to label moves into one of 4 lattice directions."""
    K = len(g); k = np.arange(K)
    c = (g * np.cos(2 * math.pi * 4 * k / K)).sum(); s = (g * np.sin(2 * math.pi * 4 * k / K)).sum()
    ph = math.atan2(s, c)                                  # in [-pi,pi], period corresponds to 90deg
    return math.cos(ph), math.sin(ph), ph


def obs_base(env, cmd, last_a):
    g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
    sh = int(round(cmd / (2 * math.pi) * KO))
    g = np.roll(g, sh); l = np.roll(l, sh)
    a = np.zeros(NACT, np.float32); a[last_a] = 1.0
    return g.astype(np.float32), l.astype(np.float32), a, np.concatenate([g, l, a, [1.0]]).astype(np.float32)


# ---------------------------------------------------------------------------------- spatial memory for 'ours'
class SpatialMemory:
    """A growing dict-of-features keyed by DRIFT-FREE topological cell coords. The policy reads a 3x3 window of
    (visited?, landmark-seen?) around the current topo cell -> size-invariant local relational context that the
    agent can build at ANY maze size (it grows with explored area; per-step read is O(1))."""
    def __init__(self): self.m = {}
    def reset(self): self.m = {}
    def mark(self, cx, cy, lm):
        e = self.m.get((cx, cy), [0.0, 0.0]); e[0] = 1.0; e[1] = max(e[1], lm); self.m[(cx, cy)] = e
    def window(self, cx, cy):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                e = self.m.get((cx + dx, cy + dy), [0.0, 0.0]); out += e
        return np.array(out, np.float32)                  # 9 cells x 2 = 18


def topo_dir(ph, last_a, moved):
    """Map the drift-free grid phase to one of 4 lattice axes for a FORWARD move. Returns a delta (dx,dy) in
    topological-cell units, or (0,0) if not a forward move. Uses the grid-mod-90 reference (allothetic) so the
    lattice coordinate does not drift with heading noise (R12: labels from the wall cue, not the metric state)."""
    if not moved: return 0, 0
    q = int(round(ph / (math.pi / 2))) % 4                # quadrant of the (drift-free) fine phase
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][q]


def odim(arch):
    if arch == "gru_raw": return GBASE
    if arch == "gru_grid": return GBASE + 2               # + (cos,sin) grid-mod-90 cue
    if arch == "transformer": return GBASE + 2
    if arch == "ours": return GBASE + 2 + 18              # grid cue + 3x3x2 spatial-memory window
    raise ValueError(arch)


# ----------------------------------------------------------------------------------------------------- models
class GRUPolicy(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(din, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True); self.pi = nn.Linear(h, NACT); self.v = nn.Linear(h, 1); self.h = h
    def init_h(self, B): return torch.zeros(1, B, self.h, device=DEV)
    def forward(self, x, hc=None):                         # x:(B,T,din)
        y, hn = self.gru(self.enc(x), hc); return self.pi(y), self.v(y).squeeze(-1), hn
    def step(self, x1, hc):                                # x1:(B,din)
        y, hn = self.gru(self.enc(x1).unsqueeze(1), hc); return self.pi(y[:, 0]), self.v(y[:, 0]).squeeze(-1), hn


class TransformerPolicy(nn.Module):
    """Causal Transformer over the egocentric stream (end-to-end baseline, no recurrent state bottleneck)."""
    def __init__(self, din, h=128, layers=2, heads=4, maxlen=4096):
        super().__init__(); self.enc = nn.Linear(din, h); self.h = h; self.maxlen = maxlen
        self.pos = nn.Embedding(maxlen, h)
        lyr = nn.TransformerEncoderLayer(h, heads, 4 * h, batch_first=True, activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(lyr, layers); self.pi = nn.Linear(h, NACT); self.v = nn.Linear(h, 1)
    def init_h(self, B): return None                       # carries the whole prefix at step time
    def _run(self, x):
        T = x.shape[1]; z = self.enc(x) + self.pos(torch.arange(T, device=DEV))[None]
        m = torch.triu(torch.full((T, T), float("-inf"), device=DEV), 1)
        y = self.tr(z, mask=m); return self.pi(y), self.v(y).squeeze(-1)
    def forward(self, x, hc=None):
        pi, v = self._run(x); return pi, v, None
    def step(self, x1, hc):                                # hc = running prefix tensor (B,t,din)
        pref = x1.unsqueeze(1) if hc is None else torch.cat([hc, x1.unsqueeze(1)], 1)
        pref = pref[:, -self.maxlen:]; pi, v = self._run(pref); return pi[:, -1], v[:, -1], pref


def build(arch, din):
    return TransformerPolicy(din).to(DEV) if arch == "transformer" else GRUPolicy(din).to(DEV)


# ------------------------------------------------------------------------------------------------- rollout
def make_obs(arch, env, cmd, last_a, mem, topo):
    g, l, a, base = obs_base(env, cmd, last_a)
    cphi, sphi, ph = grid_phase(g)
    if arch == "gru_raw": return base, ph
    if arch in ("gru_grid", "transformer"): return np.concatenate([base, [cphi, sphi]]).astype(np.float32), ph
    win = mem.window(*topo)                                # ours
    return np.concatenate([base, [cphi, sphi], win]).astype(np.float32), ph


def rollout_episode(arch, env, model, n, max_steps, step_cost, r_goal, novelty, greedy=False):
    env.reset(); cmd = 0.0; last_a = 0
    mem = SpatialMemory(); mem.reset(); topo = [0, 0]
    h = model.init_h(1)
    def cell(): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)
    seen = {cell()}
    obss, acts, logps, vals, rews = [], [], [], [], []
    reached = False
    for t in range(max_steps):
        o, ph = make_obs(arch, env, cmd, last_a, mem, tuple(topo))
        if arch == "ours":
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            mem.mark(topo[0], topo[1], 1.0 if (l > 0).any() else 0.0)
        ot = torch.tensor(o, device=DEV).unsqueeze(0)
        with torch.no_grad():
            logits, val, h = model.step(ot, h)
            probs = F.softmax(logits, -1)
            act = int(torch.argmax(probs, -1)) if greedy else int(torch.multinomial(probs, 1))
            logp = float(torch.log_softmax(logits, -1)[0, act])
        obss.append(o); acts.append(act); logps.append(logp); vals.append(float(val))
        c0 = cell()
        if act == 2: cmd -= ROT
        elif act == 3: cmd += ROT
        _, _, done, _ = env.step(act)
        last_a = act
        c1 = cell(); moved = (c1 != c0)
        if arch == "ours" and moved:                      # advance drift-free topo coord by the grid-labelled dir
            dx, dy = topo_dir(ph, act, moved); topo[0] += dx; topo[1] += dy
        rr = -step_cost
        if novelty > 0 and c1 not in seen: rr += novelty; seen.add(c1)
        if math.hypot(env.gx - env.px, env.gy - env.py) < 0.6: rr += r_goal; reached = True
        rews.append(rr)
        if reached or done: break
    return dict(obs=np.array(obss, np.float32), act=np.array(acts), logp=np.array(logps, np.float32),
                val=np.array(vals, np.float32), rew=np.array(rews, np.float32)), reached, len(acts)


ROT = 0.20


def gae(rew, val, gamma=0.99, lam=0.95):
    T = len(rew); adv = np.zeros(T, np.float32); last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]; last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + val


def evaluate(arch, model, sizes, mazes, max_mult, step_cost, r_goal, novelty, seed0=40000):
    res = {}
    for n in sizes:
        rr, st = [], []
        for m in range(mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
            _, reached, steps = rollout_episode(arch, env, model, n, max_mult * n * n, step_cost, r_goal, novelty, greedy=True)
            rr.append(1 if reached else 0); st.append(steps if reached else np.nan)
        res[n] = (float(np.mean(rr)), float(np.nanmean(st)))
    return res


def pad_batch(traj, key, pad=0.0):
    T = max(len(t[key]) for t in traj); B = len(traj)
    sh = traj[0][key].shape[1:] if traj[0][key].ndim > 1 else ()
    out = np.full((B, T) + sh, pad, np.float32)
    for i, t in enumerate(traj): out[i, :len(t[key])] = t[key]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gru_raw", choices=["gru_raw", "gru_grid", "transformer", "ours"])
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--eps_per_iter", type=int, default=32)
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[5, 7]); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--step_cost", type=float, default=0.01); ap.add_argument("--r_goal", type=float, default=5.0)
    ap.add_argument("--novelty", type=float, default=0.3); ap.add_argument("--max_mult", type=int, default=8)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28])
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval_mazes", type=int, default=20)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    din = odim(a.arch); model = build(a.arch, din); opt = torch.optim.Adam(model.parameters(), a.lr)
    rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"arch={a.arch} din={din} device={DEV} train_sizes={a.train_sizes} seed={a.seed}", flush=True)
    for it in range(a.iters):
        traj = []; reach = []; steps = []
        for e in range(a.eps_per_iter):
            n = a.train_sizes[rng.integers(len(a.train_sizes))]
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9)
            tr, rd, stp = rollout_episode(a.arch, env, model, n, a.max_mult * n * n, a.step_cost, a.r_goal, a.novelty)
            adv, ret = gae(tr["rew"], tr["val"]); tr["adv"] = adv; tr["ret"] = ret; traj.append(tr)
            reach.append(rd); steps.append(stp)
        # CORRECT recurrent PPO: pad episodes into a batch, run the model per-episode (h0=0), mask the loss.
        O = torch.tensor(pad_batch(traj, "obs"), device=DEV)
        A = torch.tensor(pad_batch(traj, "act"), device=DEV).long()
        LP = torch.tensor(pad_batch(traj, "logp"), device=DEV)
        RET = torch.tensor(pad_batch(traj, "ret"), device=DEV)
        ADV = pad_batch(traj, "adv"); MASK = np.zeros(ADV.shape, np.float32)
        for i, t in enumerate(traj): MASK[i, :len(t["adv"])] = 1.0
        valid = MASK.astype(bool)
        ADV = (ADV - ADV[valid].mean()) / (ADV[valid].std() + 1e-6)
        ADV = torch.tensor(ADV, device=DEV); M = torch.tensor(MASK, device=DEV)
        for _ in range(a.epochs):
            logits, val, _ = model(O)                      # (B,T,*) per-episode sequences -> correct ratios
            lp = torch.log_softmax(logits, -1).gather(2, A.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(lp - LP)
            l_pi = -(torch.min(ratio * ADV, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV) * M).sum() / M.sum()
            l_v = (((val - RET) ** 2) * M).sum() / M.sum()
            ent = (-(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1) * M).sum() / M.sum()
            loss = l_pi + 0.5 * l_v - 0.01 * ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (it + 1) % max(1, a.iters // 20) == 0:
            print(f"it {it+1}/{a.iters} reach={np.mean(reach):.2f} steps={np.mean(steps):.0f} "
                  f"loss={loss.item():.2f} ent={ent.item():.2f} ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== OOD eval (greedy): reach-rate / mean-steps vs size ===", flush=True)
    res = evaluate(a.arch, model, a.eval_sizes, a.eval_mazes, a.max_mult, a.step_cost, a.r_goal, a.novelty)
    for n in a.eval_sizes:
        tag = "*tr*" if n in a.train_sizes else ""
        print(f"  n={n:>3} {f'{2*n+1}x{2*n+1}':>9}{tag:>4}: reach={res[n][0]:.2f} steps={res[n][1]:.0f}", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for n in a.eval_sizes: f.write(f"{a.arch} {a.tag} {a.seed} {n} {res[n][0]:.3f} {res[n][1]:.1f}\n")
