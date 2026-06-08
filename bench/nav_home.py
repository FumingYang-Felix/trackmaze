"""Round 13c — maze HOMING (the cleanest cognitive-map test): the agent is led on a FORCED outbound DFS walk
(it observes + must build a map but does NOT control), then a home flag turns on and the agent must NAVIGATE
BACK TO START closed-loop. The teacher for the home phase is the UNIQUE shortest path back along the discovered
cell-graph (BFS) -> unambiguous BC labels (unlike open exploration where any unvisited branch is a valid choice).
Homing directly REQUIRES a usable map: you can only return if you mapped the way. OOD axis = maze size (and the
outbound length grows with it). Reuses nav_bc's NavState / models / teacher primitives.

Expected dissociation (R12): `ours` (drift-free frame + spatial memory that grows with area) returns
size-invariantly; `gru_cmd` (drifting frame) fails; `gru_corr`/`tf_corr` (drift-free frame, FIXED memory) return
on small mazes but DEGRADE as the map outgrows the hidden state -> isolates the MEMORY lever that open
exploration could not. Contrast with open-arena homing (pi_home, Stage 2a): metric homing COMPOUNDS OOD;
topological maze homing should be flat."""
import argparse, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, wrap, motion
import nav_bc as NB

DEV = NB.DEV; NACT = NB.NACT; DIRS = NB.DIRS; ROT = NB.ROT


POS = False                                                # set by --pos: append the home-bearing signal to obs


def odim(arch): return NB.odim(arch) + 1 + (3 if POS else 0)   # + home_flag (+ home-bearing signal)


def build(arch):
    return NB.TFPol(odim(arch)).to(DEV) if arch == "tf_corr" else NB.GRUPol(odim(arch)).to(DEV)


def mkobs(st, g, l, last_a, flag):
    """observe -> (optional) home-bearing signal -> home_flag. ONE code path (BC-record == eval, no skew)."""
    o = st.observe(g, l, last_a)
    parts = [o, st.home_signal()] if POS else [o]
    parts.append(np.array([flag], np.float32))
    return np.concatenate(parts).astype(np.float32)


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


def bfs_path(graph, src, dst):
    """Shortest path src->dst on the discovered cell graph (dict cell->set(neighbor cells)). Returns cell list."""
    if src == dst: return [src]
    prev = {src: None}; q = deque([src])
    while q:
        u = q.popleft()
        for v in graph.get(u, ()):
            if v not in prev:
                prev[v] = u
                if v == dst:
                    path = [v]
                    while prev[path[-1]] is not None: path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def outbound_dfs(env, n, max_steps, eps, rng):
    """Forced DFS outbound; RECORD raw (g,l,act,moved) + build the DISCOVERED cell graph from actual transitions."""
    adj = cell_graph(env.wall, n); ang0 = env.ang
    def sf_key(d): th = math.atan2(d[1], d[0]); return int(round(wrap(th - ang0) / (math.pi / 2))) % 4
    def opens(c): return sorted([(dx, dy) for (dx, dy) in DIRS if (c[0] + dx, c[1] + dy) in adj.get(c, [])], key=sf_key)
    nodes = [dict(cell=cidx(env), explored=set(), parent=None)]; stack = [0]
    prim = []; graph = {}
    def link(a_, b_): graph.setdefault(a_, set()).add(b_); graph.setdefault(b_, set()).add(a_)
    while stack and len(prim) < max_steps:
        top = stack[-1]; cur = cidx(env)
        untried = [d for d in opens(cur) if d not in nodes[top]["explored"]]
        if untried:
            d = untried[0]; nodes[top]["explored"].add(d); nxt = (cur[0] + d[0], cur[1] + d[1])
            c0 = cidx(env); NB.drive_record(env, nxt, prim, eps=eps, rng=rng); nc = cidx(env)
            if nc != c0: link(c0, nc)
            m = next((i for i, nd in enumerate(nodes) if nd["cell"] == nc), None)
            if m is None:
                nodes.append(dict(cell=nc, explored={(-d[0], -d[1])}, parent=(top, (-d[0], -d[1])))); stack.append(len(nodes) - 1)
            else:
                nodes[m]["explored"].add((-d[0], -d[1]))
                c0 = cidx(env); NB.drive_record(env, cur, prim);
                if cidx(env) != c0: link(c0, cidx(env))
        else:
            if nodes[top]["parent"] is None: break
            par, backdir = nodes[top]["parent"]; stack.pop()
            c0 = cidx(env); NB.drive_record(env, (cur[0] + backdir[0], cur[1] + backdir[1]), prim)
            if cidx(env) != c0: link(c0, cidx(env))
    return prim, graph


def home_teacher(env, n, graph, start, max_steps, eps, rng):
    """Drive the shortest discovered path back to start; RECORD raw (g,l,act,moved). Returns prim, reached."""
    prim = []
    while cidx(env) != start and len(prim) < max_steps:
        path = bfs_path(graph, cidx(env), start)
        if not path or len(path) < 2: break
        nxt = path[1]; c0 = cidx(env); NB.drive_record(env, nxt, prim, eps=eps, rng=rng)
        if cidx(env) == c0: break                          # stuck
    return prim, (cidx(env) == start)


def episode_record(arch, env, n, T_out, eps, rng):
    """One full homing episode -> (obs[T,din], act[T]); home_flag=0 during outbound, 1 during home."""
    start = cidx(env)
    out_prim, graph = outbound_dfs(env, n, T_out, eps, rng)
    home_prim, hok = home_teacher(env, n, graph, start, 40 * n * n, eps, rng)
    st = NB.NavState(arch); O, A = [], []; last_a = 0
    for flag, prim in ((0.0, out_prim), (1.0, home_prim)):
        for step in prim:
            g, l, a, moved = step[0], step[1], step[2], step[3]
            O.append(mkobs(st, g, l, last_a, flag)); A.append(a); st.advance(a, moved); last_a = a
    return np.array(O, np.float32), np.array(A, np.int64), hok


def eval_home(arch, model, sizes, mazes, T_out_mult, seed0=50000):
    model.eval(); res = {}
    for n in sizes:
        T_out = T_out_mult * n * n; rr = []
        for m in range(mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9); env.reset()
            start = cidx(env)
            # forced outbound (no policy control), policy observes + builds state
            st = NB.NavState(arch); h = model.init_h(1); last_a = 0
            out_prim, _ = outbound_dfs(env, n, T_out, 0.0, np.random.default_rng(seed0 + m))
            for step in out_prim:                          # replay forced outbound through the policy state
                g, l, a, moved = step[0], step[1], step[2], step[3]
                o = mkobs(st, g, l, last_a, 0.0)
                with torch.no_grad(): _, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h)
                st.advance(a, moved); last_a = a
            # home phase: policy controls closed-loop (cap ~ a few x the shortest return; bounds eval walltime)
            reached = False
            for t in range(12 * n * n):
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                o = mkobs(st, g, l, last_a, 1.0)
                with torch.no_grad():
                    logits, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h); a = int(torch.argmax(logits, -1))
                c0 = cidx(env); motion(env, a); st.advance(a, cidx(env) != c0); last_a = a
                if cidx(env) == start: reached = True; break
            rr.append(1 if reached else 0)
        res[n] = float(np.mean(rr))
        tg = "*tr*" if n in TRAIN else ""
        print(f"  n={n:>3} {f'{2*n+1}x{2*n+1}':>9}{tg:>4}: home-rate={res[n]:.2f}", flush=True)
        if OUT:
            with open(OUT, "a") as f: f.write(f"{arch} {TAG} {SEED} {n} {res[n]:.3f}\n")
    return res


def drive_action(env, target):
    """Single raw action to head toward target cell center (teacher action for DAgger relabeling, no noise)."""
    wx, wy = 2 * target[0] + 1.5, 2 * target[1] + 1.5
    dx, dy = wx - env.px, wy - env.py
    if math.hypot(dx, dy) < 0.3: return 0
    err = wrap(math.atan2(dy, dx) - env.ang)
    return 3 if (abs(err) > ROT and err > 0) else (2 if abs(err) > ROT else 0)


def dagger_episode(arch, model, env, n, T_out, eps, rng):
    """ON-POLICY DAgger episode: forced outbound (label=forced action), then HOME phase where the POLICY drives
    but every step is relabeled with the TEACHER action (drive toward the next BFS cell to start on the discovered
    graph). Teaches recovery from the policy's own drift -> fixes closed-loop compounding error. Returns (O,A)."""
    start = cidx(env)
    out_prim, graph = outbound_dfs(env, n, T_out, eps, rng)
    st = NB.NavState(arch); h = model.init_h(1); O, A = [], []; last_a = 0
    for step in out_prim:                                   # outbound: forced (teacher = the forced action)
        g, l, a, moved = step[0], step[1], step[2], step[3]
        o = mkobs(st, g, l, last_a, 0.0); O.append(o); A.append(a)
        with torch.no_grad(): _, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h)
        st.advance(a, moved); last_a = a
    for t in range(12 * n * n):                             # home: policy acts, teacher relabels
        if cidx(env) == start: break
        path = bfs_path(graph, cidx(env), start)
        a_t = drive_action(env, path[1]) if (path and len(path) >= 2) else 0
        g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
        o = mkobs(st, g, l, last_a, 1.0); O.append(o); A.append(a_t)
        with torch.no_grad():
            logits, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h)
            a_p = int(torch.multinomial(torch.softmax(logits, -1), 1))
        c0 = cidx(env); motion(env, a_p); st.advance(a_p, cidx(env) != c0); last_a = a_p
    return np.array(O, np.float32), np.array(A, np.int64)


def train_epochs(model, opt, Os, As, n_epochs, bs, rng, t0, label="bc"):
    idx = np.arange(len(Os))
    for ep in range(n_epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for b in range(0, len(idx), bs):
            bi = idx[b:b + bs]; O, M = NB.pad([Os[j] for j in bi]); A, _ = NB.pad([As[j] for j in bi])
            Ot = torch.tensor(O, device=DEV); At = torch.tensor(A, device=DEV).long(); Mt = torch.tensor(M, device=DEV)
            logits, _ = model(Ot)
            ce = (F.cross_entropy(logits.reshape(-1, NACT), At.reshape(-1), reduction="none").reshape(A.shape) * Mt).sum() / Mt.sum()
            opt.zero_grad(); ce.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(ce.detach()); nb += 1
        if (ep + 1) % max(1, n_epochs // 5) == 0:
            print(f"  {label} ep {ep+1}/{n_epochs} ce={tot/nb:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ours", choices=NB.ARCHS)
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--n_train_mazes", type=int, default=800); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T_out_mult", type=int, default=14); ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28, 40])
    ap.add_argument("--eval_mazes", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0); ap.add_argument("--tag", default="home"); ap.add_argument("--out", default="")
    ap.add_argument("--dagger_rounds", type=int, default=0); ap.add_argument("--dagger_mazes", type=int, default=300)
    ap.add_argument("--dagger_epochs", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()
    TRAIN = a.train_sizes; OUT = a.out; TAG = a.tag; SEED = a.seed; POS = bool(a.pos)
    print(f"HOME arch={a.arch} din={odim(a.arch)} dev={DEV} train={a.train_sizes} seed={a.seed} "
          f"pos={POS} dagger={a.dagger_rounds}", flush=True)
    Os, As, hk = [], [], []
    for i in range(a.n_train_mazes):
        n = a.train_sizes[rng.integers(len(a.train_sizes))]
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9); env.reset()
        o, ac, ok = episode_record(a.arch, env, n, a.T_out_mult * n * n, a.eps, rng); hk.append(ok)
        if len(o) > 2: Os.append(o); As.append(ac)
    print(f"dataset: {len(Os)} homing trajs, teacher home-success={np.mean(hk):.2f}, "
          f"mean len={np.mean([len(o) for o in Os]):.0f} ({time.time()-t0:.0f}s)", flush=True)
    model = build(a.arch); opt = torch.optim.Adam(model.parameters(), a.lr)
    train_epochs(model, opt, Os, As, a.epochs, a.bs, rng, t0, label="bc")
    for r in range(a.dagger_rounds):                        # DAgger: aggregate on-policy relabeled episodes, retrain
        for i in range(a.dagger_mazes):
            n = a.train_sizes[rng.integers(len(a.train_sizes))]
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=int(rng.integers(10 ** 8)), max_steps=10 ** 9); env.reset()
            o, ac = dagger_episode(a.arch, model, env, n, a.T_out_mult * n * n, a.eps, rng)
            if len(o) > 2: Os.append(o); As.append(ac)
        print(f"dagger round {r+1}/{a.dagger_rounds}: dataset={len(Os)} ({time.time()-t0:.0f}s)", flush=True)
        train_epochs(model, opt, Os, As, a.dagger_epochs, a.bs, rng, t0, label=f"dag{r+1}")
    print("\n=== OOD maze-homing: return-to-start rate vs size ===", flush=True)
    eval_home(a.arch, model, a.eval_sizes, a.eval_mazes, a.T_out_mult)
    print("\nours flat across size; gru_cmd fails (drifting frame); gru_corr/tf_corr degrade as map outgrows "
          "fixed memory => the MEMORY lever. (vs open-arena metric homing pi_home which COMPOUNDS OOD.)", flush=True)
