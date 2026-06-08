"""Round 13 (the REAL Stage-2, clean route): a LEARNED navigation policy that ACTUALLY WALKS the maze (closed
loop, egocentric obs only -- no oracle low-level execution), trained by behaviour cloning a realizable
frontier-DFS explorer (the R11/R14 teacher: reaches the exit with bounded redundancy, size-invariant, using a
visited-cell map). Eval is OOD on LARGER mazes. The spotlight claim for navigation:

  a learned policy generalizes the navigation strategy in SIZE iff it has the right inductive bias --
  (1) a DRIFT-FREE observation frame (canonicalize by the grid-corrected heading, not the drifting command
      odometry) and (2) a SPATIAL MEMORY indexed by drift-free topological coords (grows with area) --
  while end-to-end policies (drifting-frame GRU / Transformer) collapse OOD.

Why BC not RL: sparse-reward PPO on this task barely learns (reach<0.1 after 250 iters, then walltime-killed;
job 19686703) -- exploration variance swamps the arch signal. BC holds the training SIGNAL fixed (same teacher
actions, same mazes) and varies ONLY the arch -> a clean isolation of the inductive bias as the cause of OOD
transfer. At EVAL the policy drives closed-loop from egocentric obs alone ("真的走迷宫").

Archs (the contribution axis), same teacher/mazes:
  gru_cmd  : GRU, obs canonicalized by COMMAND heading (drifts), fixed memory          [e2e baseline]
  gru_corr : GRU, obs canonicalized by GRID-CORRECTED heading (drift-free) + grid cue   [frame lever]
  tf_corr  : causal Transformer, drift-free frame + grid cue                            [capacity baseline]
  ours     : GRU, drift-free frame + grid cue + spatial-memory window (drift-free topo) [+memory lever]
"""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, wrap, motion

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NACT = 4                                                   # 0 fwd, 1 back, 2 turn-left, 3 turn-right
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MV, ROT = 0.22, 0.20
GBASE = KO + KO + NACT + 1                                 # 69
ARCHS = ["gru_cmd", "gru_corr", "tf_corr", "ours", "gru_compass"]


def grid_heading(g):
    """heading mod 90deg from the freq-4 phase of the 32-ray omni signal (drift-free, allothetic ~20deg)."""
    F = np.fft.rfft(g); return float((-np.angle(F[4]) / 4.0) % (math.pi / 2))


def odim(arch):
    return {"gru_cmd": GBASE, "gru_corr": GBASE + 2, "tf_corr": GBASE + 2, "ours": GBASE + 2 + 18,
            "gru_compass": GBASE + 2}[arch]            # compass: drift-free GLOBAL heading cue (resolves quadrant)


class Mem:
    """Growing visited/landmark map keyed by drift-free topo cell -> 3x3x2 local window (size-invariant read)."""
    def __init__(self): self.m = {}
    def mark(self, c, lm):
        e = self.m.get(c, [0.0, 0.0]); e[0] = 1.0; e[1] = max(e[1], lm); self.m[c] = e
    def window(self, cx, cy):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out += self.m.get((cx + dx, cy + dy), [0.0, 0.0])
        return np.array(out, np.float32)


class NavState:
    """Per-episode running state, IDENTICAL at BC-record time and closed-loop eval time (no train/eval skew).
    Tracks command heading (cmd, drifts) and grid-corrected heading (corr, drift-free fine part) and, for
    'ours', a spatial memory addressed by topo coords derived from corr (R12: labels from the wall cue)."""
    def __init__(self, arch):
        self.arch = arch; self.cmd = 0.0; self.corr = 0.0; self.mem = Mem(); self.topo = [0, 0]; self.start = True
        self.gain = 0.3; self.pos = [0.0, 0.0]             # start-relative metric position, integrated in the arch's frame

    def observe(self, g, l, last_a, compass_h=None):
        gh = grid_heading(g)                               # drift-free heading mod 90 (start-relative via cmd's quadrant)
        gobs = (gh - 0.0) % (math.pi / 2)                  # g is agent-frame; corr is start-relative -> fuse below
        # fuse: snap corr's fine part to the grid observation, keep corr's quadrant (cmd-derived)
        k = round((self.corr - gobs) / (math.pi / 2)); target = gobs + k * (math.pi / 2)
        self.corr += self.gain * wrap(target - self.corr)
        # canonicalization heading: drifting cmd (gru_cmd) | grid-corrected (corr) | drift-free GLOBAL compass
        if self.arch == "gru_cmd": h = self.cmd
        elif self.arch == "gru_compass" and compass_h is not None: h = compass_h
        else: h = self.corr
        sh = int(round(h / (2 * math.pi) * KO)); gr = np.roll(g, sh); lr = np.roll(l, sh)
        a1 = np.zeros(NACT, np.float32); a1[last_a] = 1.0
        base = np.concatenate([gr, lr, a1, [1.0]]).astype(np.float32)
        if self.arch == "gru_cmd":
            return base
        if self.arch == "gru_compass":                     # global heading cue (resolves the Z4 quadrant at any size)
            ch = compass_h if compass_h is not None else 0.0
            cf = np.array([math.cos(ch), math.sin(ch)], np.float32)
            return np.concatenate([base, cf]).astype(np.float32)
        gf = np.array([math.cos(4 * gh), math.sin(4 * gh)], np.float32)
        if self.arch in ("gru_corr", "tf_corr"):
            return np.concatenate([base, gf]).astype(np.float32)
        self.mem.mark((self.topo[0], self.topo[1]), 1.0 if (l > 0).any() else 0.0)   # ours
        win = self.mem.window(self.topo[0], self.topo[1])
        return np.concatenate([base, gf, win]).astype(np.float32)

    def advance(self, a, moved):
        if a == 2: self.cmd -= ROT; self.corr -= ROT
        elif a == 3: self.cmd += ROT; self.corr += ROT
        if moved and a in (0, 1):                          # forward/back changed cell -> step topo + metric position
            q = int(round(self.corr / (math.pi / 2))) % 4; dx, dy = DIRS[q]
            s = 1 if a == 0 else -1; self.topo[0] += s * dx; self.topo[1] += s * dy
            h = self.cmd if self.arch == "gru_cmd" else self.corr   # integrate metric pos in the arch's OWN frame
            self.pos[0] += s * 2.0 * math.cos(h); self.pos[1] += s * 2.0 * math.sin(h)

    def home_signal(self):
        """Size-robust start-relative homing cue from the arch's OWN position estimate: (cos,sin) of the bearing
        BACK TO START relative to the current frame heading + a saturated distance. Drift-free for corr/ours,
        drifting for gru_cmd -> isolates integration quality. (The maze walls still force memory-based routing.)"""
        h = self.cmd if self.arch == "gru_cmd" else self.corr
        bear = math.atan2(-self.pos[1], -self.pos[0]) - h
        d = math.hypot(self.pos[0], self.pos[1])
        return np.array([math.cos(bear), math.sin(bear), math.tanh(d / 10.0)], np.float32)


# ---------------------------------------------------------------------------- teacher: perfect frontier DFS
def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


def drive_record(env, target, steps_prim, max_drive=80, eps=0.0, rng=None, ang0=0.0):
    """Drive toward target cell with raw actions; RECORD (g,l,act,moved,gh_true) per raw step (agent-frame omni;
    gh_true = true start-relative global heading at obs time, for the compass arch).
    eps>0 = DART noise injection: with prob eps take a RANDOM action (and record it) instead of the corrective
    one; the next iterations drive back toward target -> the dataset contains off-path states + the teacher's
    RECOVERY actions, so the BC policy learns to correct its own closed-loop drift (fixes compounding error)."""
    cx, cy = target; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
    for _ in range(max_drive):
        dx, dy = wx - env.px, wy - env.py
        if math.hypot(dx, dy) < 0.3: break
        err = wrap(math.atan2(dy, dx) - env.ang)
        a = 3 if (abs(err) > ROT and err > 0) else (2 if abs(err) > ROT else 0)
        if eps > 0 and rng is not None and rng.random() < eps: a = int(rng.integers(NACT))   # DART perturbation
        g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
        gh_true = wrap(env.ang - ang0)                     # true global heading (start-relative) at obs time
        c0 = cidx(env); motion(env, a); c1 = cidx(env)
        steps_prim.append((g.astype(np.float32), l.astype(np.float32), a, c1 != c0, gh_true))


def teacher_record(env, n, max_steps, eps=0.0, rng=None):
    """Perfect-tracker frontier DFS to the exit; returns the recorded raw-step primitives + reached.
    Untried openings are ordered in the START FRAME (using the true start heading ang0) so the teacher's choice
    is a function of what the policy OBSERVES (start-frame canonicalized) -> well-posed BC. (The absolute frame
    is unobservable, b08, so an absolute-frame priority would give the policy conflicting labels.)
    eps>0 -> DART noise injection inside drive_record (off-path recovery demos)."""
    adj = cell_graph(env.wall, n); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    ang0 = env.ang
    def sf_key(d):                                          # start-frame quadrant index of an opening dir
        th = math.atan2(d[1], d[0]); return int(round(wrap(th - ang0) / (math.pi / 2))) % 4
    def opens(c): return sorted([(dx, dy) for (dx, dy) in DIRS if (c[0] + dx, c[1] + dy) in adj.get(c, [])], key=sf_key)
    nodes = [dict(cell=cidx(env), explored=set(), parent=None)]; stack = [0]
    prim = []
    while cidx(env) != goal and stack and len(prim) < max_steps:
        top = stack[-1]; cur = cidx(env)
        untried = [d for d in opens(cur) if d not in nodes[top]["explored"]]
        if untried:
            d = untried[0]; nodes[top]["explored"].add(d); nxt = (cur[0] + d[0], cur[1] + d[1])
            drive_record(env, nxt, prim, eps=eps, rng=rng, ang0=ang0); nc = cidx(env)
            m = next((i for i, nd in enumerate(nodes) if nd["cell"] == nc), None)
            if m is None:
                nodes.append(dict(cell=nc, explored={(-d[0], -d[1])}, parent=(top, (-d[0], -d[1])))); stack.append(len(nodes) - 1)
            else:
                nodes[m]["explored"].add((-d[0], -d[1])); drive_record(env, cur, prim, eps=eps, rng=rng, ang0=ang0)   # loop -> retreat
        else:
            if nodes[top]["parent"] is None: break
            par, backdir = nodes[top]["parent"]; stack.pop()
            drive_record(env, (cur[0] + backdir[0], cur[1] + backdir[1]), prim, eps=eps, rng=rng, ang0=ang0)
    return prim, (cidx(env) == goal)


def build_seq(arch, prim):
    """Replay recorded primitives through NavState -> (obs[T,din], act[T]) for this arch (no env needed)."""
    st = NavState(arch); O, A = [], []; last_a = 0
    for step in prim:
        g, l, a, moved = step[0], step[1], step[2], step[3]; gh = step[4] if len(step) > 4 else None
        O.append(st.observe(g, l, last_a, compass_h=gh)); A.append(a); st.advance(a, moved); last_a = a
    return np.array(O, np.float32), np.array(A, np.int64)


# ------------------------------------------------------------------------------------------------- models
class GRUPol(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(din, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True); self.pi = nn.Linear(h, NACT); self.h = h
    def init_h(self, B): return torch.zeros(1, B, self.h, device=DEV)
    def forward(self, x, hc=None): y, hn = self.gru(self.enc(x), hc); return self.pi(y), hn
    def step(self, x1, hc): y, hn = self.gru(self.enc(x1).unsqueeze(1), hc); return self.pi(y[:, 0]), hn


class TFPol(nn.Module):
    def __init__(self, din, h=128, layers=2, heads=4, maxlen=8192):
        super().__init__(); self.enc = nn.Linear(din, h); self.pos = nn.Embedding(maxlen, h); self.maxlen = maxlen
        lyr = nn.TransformerEncoderLayer(h, heads, 4 * h, batch_first=True, activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(lyr, layers); self.pi = nn.Linear(h, NACT)
    def init_h(self, B): return None
    def _run(self, x):
        T = x.shape[1]; z = self.enc(x) + self.pos(torch.arange(T, device=DEV))[None]
        m = torch.triu(torch.full((T, T), float("-inf"), device=DEV), 1); return self.pi(self.tr(z, mask=m))
    def forward(self, x, hc=None): return self._run(x), None
    def step(self, x1, hc):
        pref = x1.unsqueeze(1) if hc is None else torch.cat([hc, x1.unsqueeze(1)], 1)
        pref = pref[:, -self.maxlen:]; return self._run(pref)[:, -1], pref


def build(arch): return TFPol(odim(arch)).to(DEV) if arch == "tf_corr" else GRUPol(odim(arch)).to(DEV)


# ------------------------------------------------------------------------------------------------- eval
def eval_closed(arch, model, sizes, mazes, max_mult, seed0=40000, train_sizes=(), out="", tag="", seed=0):
    model.eval(); res = {}
    for n in sizes:
        rr, ratio = [], []
        for m in range(mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); env.reset()
            ang0 = env.ang                                  # for the compass arch's drift-free global cue
            # teacher steps on the SAME maze (for the redundancy ratio)
            te = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); te.reset()
            tprim, tok = teacher_record(te, n, max_mult * n * n)
            st = NavState(arch); h = model.init_h(1); last_a = 0; reached = False; T = max_mult * n * n
            for t in range(T):
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                o = st.observe(g, l, last_a, compass_h=wrap(env.ang - ang0)); ot = torch.tensor(o, device=DEV).unsqueeze(0)
                with torch.no_grad():
                    logits, h = model.step(ot, h); a = int(torch.argmax(logits, -1))
                c0 = cidx(env); motion(env, a); c1 = cidx(env); st.advance(a, c1 != c0); last_a = a
                if math.hypot(env.gx - env.px, env.gy - env.py) < 0.6: reached = True; break
            rr.append(1 if reached else 0)
            ratio.append((t + 1) / max(1, len(tprim)) if (reached and tok) else np.nan)
        res[n] = (float(np.mean(rr)), float(np.nanmean(ratio)))
        tg = "*tr*" if n in train_sizes else ""
        print(f"  n={n:>3} {f'{2*n+1}x{2*n+1}':>9}{tg:>4}: reach={res[n][0]:.2f} ratio={res[n][1]:.2f}", flush=True)
        if out:                                            # write incrementally so a timeout keeps partial results
            with open(out, "a") as f: f.write(f"{arch} {tag} {seed} {n} {res[n][0]:.3f} {res[n][1]:.2f}\n")
    return res


def pad(seqs, pad_val=0.0):
    T = max(len(s) for s in seqs); B = len(seqs); sh = seqs[0].shape[1:] if seqs[0].ndim > 1 else ()
    out = np.full((B, T) + sh, pad_val, np.float32 if seqs[0].dtype != np.int64 else np.int64)
    msk = np.zeros((B, T), np.float32)
    for i, s in enumerate(seqs): out[i, :len(s)] = s; msk[i, :len(s)] = 1.0
    return out, msk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ours", choices=ARCHS)
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--n_train_mazes", type=int, default=600); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_mult", type=int, default=10); ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28, 40])
    ap.add_argument("--eval_mazes", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", default=""); ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"arch={a.arch} din={odim(a.arch)} dev={DEV} train_sizes={a.train_sizes} seed={a.seed}", flush=True)
    # ---- build the BC dataset (teacher trajectories on small mazes) ----
    Os, As, tr_reach = [], [], []
    for i in range(a.n_train_mazes):
        n = a.train_sizes[rng.integers(len(a.train_sizes))]
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9); env.reset()
        prim, ok = teacher_record(env, n, a.max_mult * n * n, eps=a.eps, rng=rng); tr_reach.append(ok)
        if len(prim) < 2: continue
        o, ac = build_seq(a.arch, prim); Os.append(o); As.append(ac)
    print(f"dataset: {len(Os)} teacher trajs, teacher reach={np.mean(tr_reach):.2f}, "
          f"mean len={np.mean([len(o) for o in Os]):.0f} ({time.time()-t0:.0f}s)", flush=True)
    model = build(a.arch); opt = torch.optim.Adam(model.parameters(), a.lr)
    idx = np.arange(len(Os))
    for ep in range(a.epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for b in range(0, len(idx), a.bs):
            bi = idx[b:b + a.bs]
            O, M = pad([Os[j] for j in bi]); A, _ = pad([As[j] for j in bi])
            Ot = torch.tensor(O, device=DEV); At = torch.tensor(A, device=DEV).long(); Mt = torch.tensor(M, device=DEV)
            logits, _ = model(Ot)
            ce = (F.cross_entropy(logits.reshape(-1, NACT), At.reshape(-1), reduction="none").reshape(A.shape) * Mt).sum() / Mt.sum()
            opt.zero_grad(); ce.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(ce); nb += 1
        if (ep + 1) % max(1, a.epochs // 10) == 0:
            print(f"ep {ep+1}/{a.epochs} bc_ce={tot/nb:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if a.save:
        torch.save({"state_dict": model.state_dict(), "arch": a.arch, "din": odim(a.arch)}, a.save)
        print(f"saved checkpoint -> {a.save}", flush=True)
    print("\n=== OOD closed-loop eval: reach-rate / steps-vs-teacher ratio (lower=better, ~1=optimal) ===", flush=True)
    res = eval_closed(a.arch, model, a.eval_sizes, a.eval_mazes, a.max_mult,
                      train_sizes=a.train_sizes, out=a.out, tag=a.tag, seed=a.seed)
    print("\nours: reach~1 & ratio flat across size; gru_cmd collapses OOD (drifting frame); "
          "gru_corr/tf_corr in between => structure (drift-free frame + spatial memory) drives OOD navigation.", flush=True)
