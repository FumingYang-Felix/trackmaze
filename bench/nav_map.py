"""STAGE-2 'B': a navigation policy with a COORDINATE-ADDRESSED GROWING MAP (the redesign of the MGM ring buffer).
Memory = a sparse grid keyed by DRIFT-FREE topo coords (Stage-1's grid-corrected frame), each cell storing
{visited, exits_explored[4], parent_dir} -- a RICHER map than nav_bc's 'ours' (visited-only window), giving the
policy the structure for the 3 egocentric-nav operations: (1) circle-back = coord already visited; (2) dead-end
backtrack = parent_dir; (3) frontier select = an OPEN exit not yet explored.

We REUSE nav_bc's proven teacher (good DFS demonstrations) and frame logic; only the per-step STATE/OBS is enriched
(MapState below mirrors NavState's drift-free cmd/corr/topo update and adds the map). The map window is in START-FRAME
(consistent with the canonicalized obs). BC the policy, run closed-loop OOD-size. The only size-coupled failure is
topo-coord drift (the Z4 quadrant slip = the same limit the deployable-SLAM work bounds). Milestone B (clean
size-invariant nav); C will make the map/planner learned end-to-end."""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import wrap, motion
import nav_bc as NB

DEV = NB.DEV; NACT = NB.NACT; ROT = NB.ROT; DIRS = NB.DIRS; GBASE = NB.GBASE
MAPW = 12                                                  # map window: exits(4)+neighbor-visited(4)+parent-onehot(4)


def odim(): return GBASE + 2 + MAPW                        # canonicalized g,l,a,bias(69) + grid cue(2) + map(12) = 83


class TopoMap:
    """(x,y) -> [visited, e0..e3 (exits explored, START-frame dir order = DIRS), parent_dir (-1 none)]."""
    def __init__(self): self.m = {(0, 0): [1.0, 0, 0, 0, 0, -1.0]}
    def rec(self, c): return self.m.get(c, [0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    def visited(self, c): return self.m.get(c, [0])[0] > 0
    def visit(self, c, parent):
        if c not in self.m: self.m[c] = [1.0, 0, 0, 0, 0, float(parent if parent is not None else -1)]
    def mark_exit(self, c, d):
        r = self.m.get(c) or [0.0, 0, 0, 0, 0, -1.0]; r[1 + d] = 1.0; self.m[c] = r


class MapState:
    """Drift-free cmd/corr/topo (mirrors nav_bc.NavState) + the coordinate-addressed TopoMap. Identical at
    teacher-record and eval -> no train/eval skew. Window is START-frame (matches the corr-canonicalized obs)."""
    def __init__(self):
        self.cmd = 0.0; self.corr = 0.0; self.gain = 0.3; self.topo = (0, 0); self.map = TopoMap()

    def observe(self, g, l, last_a):
        gh = NB.grid_heading(g)
        k = round((self.corr - (gh % (math.pi / 2))) / (math.pi / 2)); tgt = (gh % (math.pi / 2)) + k * (math.pi / 2)
        self.corr += self.gain * wrap(tgt - self.corr)
        sh = int(round(self.corr / (2 * math.pi) * KO)); gr = np.roll(g, sh); lr = np.roll(l, sh)
        a1 = np.zeros(NACT, np.float32); a1[last_a] = 1.0
        gf = np.array([math.cos(4 * gh), math.sin(4 * gh)], np.float32)
        rec = self.map.rec(self.topo)
        exits = [rec[1 + d] for d in range(4)]                                   # explored exits (start-frame)
        nbr = [1.0 if self.map.visited((self.topo[0] + DIRS[d][0], self.topo[1] + DIRS[d][1])) else 0.0 for d in range(4)]
        par = [1.0 if int(rec[5]) == d else 0.0 for d in range(4)]               # which dir is the way-back
        win = np.array(exits + nbr + par, np.float32)
        return np.concatenate([gr, lr, a1, [1.0], gf, win]).astype(np.float32)

    def advance(self, a, moved):
        if a == 2: self.cmd -= ROT; self.corr -= ROT
        elif a == 3: self.cmd += ROT; self.corr += ROT
        if moved and a == 0:                                                      # forward into a (new/old) cell
            d = int(round(self.corr / (math.pi / 2))) % 4                         # start-frame dir moved (drift-free estimate)
            self.map.mark_exit(self.topo, d)
            nc = (self.topo[0] + DIRS[d][0], self.topo[1] + DIRS[d][1])
            self.map.visit(nc, (d + 2) % 4); self.topo = nc


def build_seq(prim):
    """Replay nav_bc teacher primitives (g,l,a,moved,gh) through MapState -> (obs[T,din], act[T])."""
    st = MapState(); O, A = [], []; last_a = 0
    for step in prim:
        g, l, a, moved = step[0], step[1], step[2], step[3]
        O.append(st.observe(g, l, last_a)); A.append(a); st.advance(a, moved); last_a = a
    return np.array(O, np.float32), np.array(A, np.int64)


def eval_closed(model, sizes, mazes, max_mult, seed0=40000, train_sizes=(), out="", seed=0):
    model.eval(); res = {}
    for n in sizes:
        rr, ratio = [], []
        for m in range(mazes):
            te = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); te.reset()
            tprim, tok = NB.teacher_record(te, n, max_mult * n * n)
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); env.reset()
            st = MapState(); h = model.init_h(1); last_a = 0; reached = False; T = max_mult * n * n
            for t in range(T):
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                o = st.observe(g, l, last_a)
                with torch.no_grad():
                    logits, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h); a = int(torch.argmax(logits, -1))
                c0 = ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2); motion(env, a)
                c1 = ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2); st.advance(a, c1 != c0); last_a = a
                if math.hypot(env.gx - env.px, env.gy - env.py) < 0.6: reached = True; break
            rr.append(1 if reached else 0); ratio.append((t + 1) / max(1, len(tprim)) if (reached and tok) else np.nan)
        res[n] = (float(np.mean(rr)), float(np.nanmean(ratio)))
        tg = "*tr*" if n in train_sizes else ""
        print(f"  n={n:>3} {f'{2*n+1}x{2*n+1}':>9}{tg:>4}: reach={res[n][0]:.2f} ratio={res[n][1]:.2f}", flush=True)
        if out:
            with open(out, "a") as f: f.write(f"ours_map - {seed} {n} {res[n][0]:.3f} {res[n][1]:.2f}\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--n_train_mazes", type=int, default=800); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_mult", type=int, default=25); ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28, 40])
    ap.add_argument("--eval_mazes", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=""); ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"NAV-MAP (B) din={odim()} dev={DEV} train={a.train_sizes} seed={a.seed}", flush=True)
    Os, As, tr = [], [], []
    for i in range(a.n_train_mazes):
        n = a.train_sizes[rng.integers(len(a.train_sizes))]
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9); env.reset()
        prim, ok = NB.teacher_record(env, n, a.max_mult * n * n, eps=a.eps, rng=rng); tr.append(ok)
        if len(prim) > 2: o, ac = build_seq(prim); Os.append(o); As.append(ac)
    print(f"dataset: {len(Os)} trajs, teacher reach={np.mean(tr):.2f}, mean len={np.mean([len(o) for o in Os]):.0f} ({time.time()-t0:.0f}s)", flush=True)
    model = NB.GRUPol(odim()).to(DEV); opt = torch.optim.Adam(model.parameters(), a.lr); idx = np.arange(len(Os))
    for ep in range(a.epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for b in range(0, len(idx), a.bs):
            bi = idx[b:b + a.bs]; O, M = NB.pad([Os[j] for j in bi]); A, _ = NB.pad([As[j] for j in bi])
            Ot = torch.tensor(O, device=DEV); At = torch.tensor(A, device=DEV).long(); Mt = torch.tensor(M, device=DEV)
            logits, _ = model(Ot)
            ce = (F.cross_entropy(logits.reshape(-1, NACT), At.reshape(-1), reduction="none").reshape(A.shape) * Mt).sum() / Mt.sum()
            opt.zero_grad(); ce.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(ce.detach()); nb += 1
        if (ep + 1) % max(1, a.epochs // 8) == 0: print(f"ep {ep+1}/{a.epochs} ce={tot/nb:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if a.save:
        torch.save({"state_dict": model.state_dict(), "din": odim()}, a.save); print(f"saved -> {a.save}", flush=True)
    print("\n=== OOD closed-loop: reach / steps-vs-teacher ratio ===", flush=True)
    eval_closed(model, a.eval_sizes, a.eval_mazes, a.max_mult, train_sizes=a.train_sizes, out=a.out, seed=a.seed)
    print("\nrich coordinate-addressed map (exits+parent) should reach further/cleaner than 'ours' (visited-only).", flush=True)
