"""Round 3-A (training-free mechanism test): WHY does GLOBAL drift grow with size, and can a BOUNDED,
importance-weighted memory match an UNBOUNDED one (= can compression keep global tracking size-invariant)?

Exploration is driven by an ORACLE DFS walk over the true maze graph (depth-first with backtracking) so the
agent COVERS the whole maze and genuinely REVISITS earlier cells (loop closures) -- this is what makes the
explored area, and hence the place memory, grow with size. (The scripted forward-explorer spins in place and
covers ~6 cells, so it can never stress memory.) Exploration uses ground truth; TRACKING does not: the agent
integrates only its commanded turns/steps -> a drifting estimate of displacement-from-start (the env injects
unobservable odometry + persistent heading noise + collisions, exactly R2-C's GLOBAL error).

LOOP CLOSURE: on revisiting a stored cell (oracle place identity, to isolate the MEMORY question from the
recognition question) we snap the estimate toward the stored, earlier (lower-drift) estimate of that cell.
Backtracking to shallow near-start cells (tiny drift) is what can bound global drift.

Variants:
  none      : pure dead-reckoning (baseline; global grows) -- is it just drift?
  unbounded : store every visited cell (loop closure with O(cells-visited) = O(n^2) memory)
  fifo  K   : bounded K, evict oldest (uniform compression -- throws away old near-start anchors)
  imp   K   : bounded K, evict fewest-revisits (keep cells you keep coming back to)
  near  K   : bounded K, evict farthest-estimate (keep low-drift near-start anchors = most "retraceable")

Decisive reads:
  unbounded FLAT while none GROWS, AND unbounded mem grows with n => global drift IS loop-closure-correctable
  but the memory it needs grows => compression is the lever. imp/near K (K << n^2) tracking unbounded while
  fifo degrades => importance-weighted compression keeps global tracking size-invariant with sublinear memory.
"""
import argparse, math, sys, numpy as np
from env import TrackMazeEnv

sys.setrecursionlimit(2_000_000)
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def cell_graph(wall, n):
    adj = {}
    for cy in range(n):
        for cx in range(n):
            nb = []
            for dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n and 0 <= ny < n and wall[2 * cy + 1 + dy, 2 * cx + 1 + dx] == 0:
                    nb.append((nx, ny))
            adj[(cx, cy)] = nb
    return adj


def dfs_walk(adj, start=(0, 0)):
    """Depth-first tour: each cell appended on entry, parent re-appended on backtrack -> revisits."""
    walk, visited = [], set()
    stack = [(start, 0)]; visited.add(start); walk.append(start)
    while stack:
        u, i = stack[-1]
        nbrs = adj[u]
        advanced = False
        while i < len(nbrs):
            v = nbrs[i]; i += 1
            if v not in visited:
                visited.add(v); stack[-1] = (u, i); stack.append((v, 0)); walk.append(v); advanced = True; break
        if not advanced:
            stack.pop()
            if stack: walk.append(stack[-1][0])   # backtrack to parent
    return walk


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion(env, action):
    """Replicates env.step's motion (collision + odometry/heading noise) WITHOUT raycast (fast)."""
    mv, rot = 0.22, 0.20
    if action == 2:   env.ang -= rot
    elif action == 3: env.ang += rot
    else:
        f = mv if action == 0 else -mv
        if env.action_noise: f += float(env.rng.normal(0, env.action_noise))
        nx = env.px + math.cos(env.ang) * f; ny = env.py + math.sin(env.ang) * f
        if env.wall[int(env.py), int(nx)] == 0: env.px = nx
        if env.wall[int(ny), int(env.px)] == 0: env.py = ny
    if env.rot_noise: env.ang += float(env.rng.normal(0, env.rot_noise))


def run(n, variant, K, gain=1.0, n_mazes=15, wp_budget=40):
    mv, rot = 0.22, 0.20
    fin, mean_e, p90_e, mems, ncells = [], [], [], [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=3000 + m, max_steps=10 ** 9)
        env.reset()
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
        true0 = np.array([env.px, env.py], dtype=np.float64)
        est = np.zeros(2); ang_est = env.ang             # start heading known (gauge), tracked via commands
        mem = {}; maxmem = 0; step_err = []
        for (cx, cy) in walk[1:]:
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5          # target cell center (world coords)
            for _ in range(wp_budget):
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                err = wrap(math.atan2(dy, dx) - env.ang)
                if abs(err) > rot:
                    a = 3 if err > 0 else 2              # turn toward waypoint
                    if a == 2: ang_est -= rot
                    else:      ang_est += rot
                else:
                    a = 0                               # step forward
                    est = est + np.array([math.cos(ang_est), math.sin(ang_est)]) * mv
                motion(env, a)
                cell = (int(env.px), int(env.py))
                if cell in mem:
                    stored = mem[cell]
                    est = est + gain * (stored[0] - est)            # loop closure
                    stored[1] += 1; stored[2] = len(mem)
                else:
                    if variant != "none":
                        if variant in ("fifo", "imp", "near") and len(mem) >= K:
                            if variant == "fifo":   victim = min(mem, key=lambda c: mem[c][2])
                            elif variant == "imp":  victim = min(mem, key=lambda c: (mem[c][1], mem[c][2]))
                            else:                   victim = max(mem, key=lambda c: float(mem[c][0] @ mem[c][0]))
                            del mem[victim]
                        mem[cell] = [est.copy(), 1, len(mem)]
                maxmem = max(maxmem, len(mem))
                step_err.append(float(np.linalg.norm(est - (np.array([env.px, env.py]) - true0))))
        fin.append(step_err[-1]); mean_e.append(float(np.mean(step_err)))
        p90_e.append(float(np.percentile(step_err, 90))); mems.append(maxmem)
        ncells.append(len(set(walk)))
    return (float(np.mean(fin)), float(np.mean(mean_e)), float(np.mean(p90_e)),
            float(np.mean(mems)), float(np.mean(ncells)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=15)
    a = ap.parse_args()
    variants = [("none", 0), ("unbounded", 0), ("fifo", a.K), ("imp", a.K), ("near", a.K)]
    print(f"GLOBAL error vs size (K={a.K}, {a.mazes} mazes, loop=0.3, oracle DFS explore). "
          f"Reporting MEAN error over the WHOLE trajectory (not just final).")
    print(f"{'n':>4} {'grid':>8} " + " ".join(f"{v[0]:>10}" for v in variants) + f" | {'cells':>6} {'ubnd_mem':>8}")
    grow = {v[0]: [] for v in variants}
    detail = {}
    for n in a.sizes:
        row = []; cells = mem_u = 0
        for name, K in variants:
            fin, mean_e, p90, mm, nc = run(n, name, K, n_mazes=a.mazes)
            row.append(mean_e); grow[name].append(mean_e); detail[(n, name)] = (fin, mean_e, p90)
            if name == "unbounded": mem_u = mm; cells = nc
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} " + " ".join(f"{e:10.2f}" for e in row) + f" | {cells:6.0f} {mem_u:8.0f}")
    print("\nGROWTH of MEAN trajectory error (n_min -> n_max):")
    for name, _ in variants:
        g = grow[name]; print(f"  {name:>10}: {g[0]:.2f} -> {g[-1]:.2f}  ({g[-1]-g[0]:+.2f})")
    print("\nfinal / mean / p90 error at largest size:")
    nL = a.sizes[-1]
    for name, _ in variants:
        f, mu, p = detail[(nL, name)]; print(f"  {name:>10}: final={f:6.2f}  mean={mu:6.2f}  p90={p:6.2f}")
