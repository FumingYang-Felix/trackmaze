"""Round 11 (A — navigation): does the LEARNED tracker (MGM) enable the user's goal -- reach the exit
WITHOUT getting lost (no redundant re-walking) -- size-invariantly, where raw odometry (R7-C) failed (0%)?

The agent does DFS exploration to the exit. Low-level corridor traversal is oracle (env adjacency, like R7-C);
the DECISION layer uses an online state estimate for LOOP DETECTION: at each newly-entered cell, if the
estimate says "I'm within radius r of a place I've already mapped" (and it's not the DFS parent), treat it as
a loop -> link + retreat instead of re-descending. The estimate source:
  perfect : true cell (ceiling -- pure DFS, no re-exploration)
  odo     : raw command-integrated odometry (R7-C: failed, drift -> false loops -> skips the exit's branch)
  mgm     : the trained MGM run ONLINE (its displacement estimate mu)  <-- the test
The MGM is fed exactly as in training (generate_allo order): per env action, omni view rolled by the
command-integrated heading + commanded step in the start frame, stepped through model.step.

Metric: success (reach exit) + steps relative to the perfect-map DFS, vs size. If mgm ~ perfect (succeeds,
low redundancy, size-invariant) while odo fails, the learned tracker delivers the navigation goal.
"""
import argparse, math, numpy as np, torch
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, wrap, motion
from arch_mgm import build

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MV, ROT = 0.22, 0.20


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


class Tracker:
    """Online position estimate fed by per-env-action obs+motion, matching generate_allo(canon='cmd')."""
    def __init__(self, env, ident, model):
        self.ident, self.model = ident, model
        self.ang0 = env.ang; self.cmd = 0.0; self.mu = np.zeros(2)
        self.st = model.init_state(1, "cpu") if ident == "mgm" else None

    def act(self, env, a):
        """Call BEFORE env.step(a): consume the obs at the current pose + the commanded motion."""
        if self.ident == "perfect":
            if a == 2: self.cmd -= ROT
            elif a == 3: self.cmd += ROT
            return
        f = MV if a == 0 else (-MV if a == 1 else 0.0)
        mo = np.array([f * math.cos(self.cmd), f * math.sin(self.cmd)], np.float32)
        if self.ident == "odo":
            self.mu = self.mu + mo
        else:  # mgm
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            sh = int(round(self.cmd / (2 * math.pi) * KO))
            g = np.roll(g, sh); l = np.roll(l, sh)
            with torch.no_grad():
                mu = self.model.step(torch.tensor(g[None], dtype=torch.float32),
                                     torch.tensor(l[None], dtype=torch.float32),
                                     torch.tensor(mo[None]), self.st)
            self.mu = mu[0].numpy()
        if a == 2: self.cmd -= ROT
        elif a == 3: self.cmd += ROT


def drive(env, tracker, target_cell, counter):
    cx, cy = target_cell; wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
    for _ in range(80):
        dx, dy = wx - env.px, wy - env.py
        if math.hypot(dx, dy) < 0.3: break
        err = wrap(math.atan2(dy, dx) - env.ang)
        a = 3 if (abs(err) > ROT and err > 0) else (2 if abs(err) > ROT else 0)
        tracker.act(env, a); motion(env, a)
    counter[0] += 1


def navigate(env, n, ident, model, r=1.2):
    adj = cell_graph(env.wall, n)
    start = cidx(env); goal = ((int(env.gx) - 1) // 2, (int(env.gy) - 1) // 2)
    tr = Tracker(env, ident, model)
    nodes = []   # dict(repcell, est, explored=set, parent=(idx,backdir))

    def opens(cell): return [(dx, dy) for (dx, dy) in DIRS if (cell[0] + dx, cell[1] + dy) in adj.get(cell, [])]

    def match(cur_node, cell, est):
        if ident == "perfect":
            for i, nd in enumerate(nodes):
                if nd["repcell"] == cell: return i
            return None
        best, bd = None, r * r
        for i, nd in enumerate(nodes):
            d = est - nd["est"]; dd = float(d @ d)
            if dd < bd: bd, best = dd, i
        return best

    nodes.append(dict(repcell=start, est=tr.mu.copy(), explored=set(), parent=None))
    stack = [0]; steps = [0]; visited = {start}; safety = 80 * n * n
    while cidx(env) != goal and stack and steps[0] < safety:
        top = stack[-1]; cur = cidx(env)
        untried = [d for d in opens(cur) if d not in nodes[top]["explored"]]
        if untried:
            d = untried[0]; nodes[top]["explored"].add(d)
            nxt = (cur[0] + d[0], cur[1] + d[1])
            drive(env, tr, nxt, steps)
            nc = cidx(env); visited.add(nc)
            m = match(top, nc, tr.mu)
            if m is None:
                nid = len(nodes)
                nodes.append(dict(repcell=nc, est=tr.mu.copy(), explored={(-d[0], -d[1])}, parent=(top, (-d[0], -d[1]))))
                stack.append(nid)
            else:
                nodes[m]["explored"].add((-d[0], -d[1])); drive(env, tr, cur, steps)   # loop -> retreat
        else:
            if nodes[top]["parent"] is None: break
            par, backdir = nodes[top]["parent"]; stack.pop()
            drive(env, tr, (cur[0] + backdir[0], cur[1] + backdir[1]), steps)
    return cidx(env) == goal, steps[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mgm_v2_big_s0.pt"); ap.add_argument("--arch", default="mgm_v2_big")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=6); ap.add_argument("--r", type=float, default=1.2)
    a = ap.parse_args()
    model = build(a.arch); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model.eval()
    print(f"Round 11 navigation: reach exit without getting lost. ckpt={a.ckpt} r={a.r}")
    print(f"{'n':>4} {'grid':>8} | " + " | ".join(f"{i:>20}" for i in ["perfect", "odo", "mgm"]))
    print(f"{'':>4} {'':>8} | " + " | ".join(f"{'succ steps ratio':>20}" for _ in range(3)))
    for n in a.sizes:
        agg = {k: dict(s=[], st=[]) for k in ["perfect", "odo", "mgm"]}
        for m in range(a.mazes):
            for ident in ["perfect", "odo", "mgm"]:
                env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=18000 + m, max_steps=10 ** 9); env.reset()
                ok, st = navigate(env, n, ident, model, a.r)
                agg[ident]["s"].append(1 if ok else 0); agg[ident]["st"].append(st if ok else np.nan)
        pst = np.nanmean(agg["perfect"]["st"])
        segs = []
        for k in ["perfect", "odo", "mgm"]:
            sc = np.mean(agg[k]["s"]); stp = np.nanmean(agg[k]["st"])
            segs.append(f"{sc:4.2f} {stp:6.0f} {stp/pst if pst and stp==stp else 0:5.1f}")
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | " + " | ".join(segs))
    print("\nmgm succeeds with ratio ~1 & flat (like perfect) while odo fails => the learned tracker enables")
    print("size-invariant navigation (reach exit, don't get lost) where raw odometry cannot.")
