"""Demo video with a REAL model's belief (not hand-coded): a trained GRU path-integrator walks a maze;
its position estimate (decoded from its latent) drifts away from the truth. Honest: the model does NOT
re-anchor on landmarks (our phase-1 finding), so the belief keeps drifting even past landmarks.

Left: the actual maze (walls + colored landmarks), true path + true dot (white), the model's believed
position (amber) + drift line. Saves maze_belief.gif + a 3-frame strip.
"""
import math, colorsys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from env import TrackMazeEnv
from generate import explore_policy, egocentric_disp
from train_eval import cached, encode, train, AMB
from archs import build

torch.manual_seed(0); np.random.seed(0)
print("training the GRU integrator (landmark-free, the real model) ...")
tr = cached(f"train_n6_T160_a{AMB}", n=6, n_eps=200, T=160, ambiguity=AMB, seed=1)
x, y = encode(tr, use_landmarks=False); model = build("gru", x.shape[2]); train(model, x, y, 60, "cpu")

def maze_graph(env, n):
    adj = {}
    for j in range(n):
        for i in range(n):
            nb = []
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ni, nj = i+di, j+dj
                if 0 <= ni < n and 0 <= nj < n and env.wall[2*j+1+dj][2*i+1+di] == 0: nb.append((ni, nj))
            adj[(i, j)] = nb
    return adj

def dfs_walk(adj, start, rng, maxlen):                       # DFS over cells with backtracking (revisits)
    path = [start]; vis = {start}; stack = [start]
    while stack and len(path) < maxlen:
        cur = stack[-1]; unv = [c for c in adj[cur] if c not in vis]
        if unv:
            nxt = unv[int(rng.integers(len(unv)))]; vis.add(nxt); stack.append(nxt); path.append(nxt)
        else:
            stack.pop()
            if stack: path.append(stack[-1])
    return path

def rollout(env, n, T, seed):                                # waypoint-follow a DFS traversal of the maze
    rng = np.random.default_rng(seed); path = dfs_walk(maze_graph(env, n), (0, 0), rng, maxlen=T)
    wps = [(2*i+1.5, 2*j+1.5) for (i, j) in path]
    obs = env.reset(); ang0 = env.ang; wi = 1
    feat, lm, po = [], [], [np.array([env.px, env.py], np.float32)]
    for t in range(T):
        tx, ty = wps[min(wi, len(wps)-1)]
        des = math.atan2(ty - env.py, tx - env.px); err = (des - env.ang + math.pi) % (2*math.pi) - math.pi
        a = 0 if abs(err) < 0.18 else (3 if err > 0 else 2)
        feat.append(np.concatenate([obs["rays"], obs["last_action"]]).astype(np.float32))
        lm.append(obs["ray_lm"].astype(np.int32)); obs, _, _, gt = env.step(a); po.append(gt["pos"])
        if math.hypot(tx - env.px, ty - env.py) < 0.35 and wi < len(wps)-1: wi += 1
    pos = np.stack(po[:T]); cov = len({(int(p[0]), int(p[1])) for p in pos})
    return pos, np.stack(feat), np.stack(lm), ang0, cov

print("rolling out a maze traversal ...")
N, T = 8, 420
env = TrackMazeEnv(n=N, ambiguity=AMB, seed=4, max_steps=T+2)
pos, feat, lm, ang0, cov = rollout(env, N, T, seed=4)
print(f"  traversal covers {cov} cells")
ds = dict(feat=feat[None], ray_lm=lm[None], disp=egocentric_disp(pos, ang0)[None])
xb, _ = encode(ds, use_landmarks=False)
with torch.no_grad(): ego = model(xb)[0].numpy()             # model's egocentric displacement estimate
c, s = math.cos(ang0), math.sin(ang0)                        # rotate ego disp back to world frame
bel = pos[0] + np.stack([c*ego[:,0] - s*ego[:,1], s*ego[:,0] + c*ego[:,1]], 1)
drift = np.linalg.norm(bel - pos, axis=1)

# maze image (walls + landmarks)
W = env.wall.shape[0]; img = np.zeros((W, W, 3))
for yy in range(W):
    for xx in range(W):
        if env.wall[yy, xx] == 0: img[yy, xx] = (0.10, 0.11, 0.15)
        elif env.col[yy, xx] > 0:
            h = ((env.col[yy, xx]-1) / max(1, env.LBINS)); img[yy, xx] = colorsys.hsv_to_rgb(h, 0.7, 0.85)
        else: img[yy, xx] = (0.23, 0.25, 0.30)

def frame(ax, t):
    ax.clear(); ax.imshow(img, extent=[0, W, W, 0], interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(W, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.plot(pos[:t+1,0], pos[:t+1,1], "-", color="#4c9be8", lw=1, alpha=0.5)
    ax.plot([pos[t,0], bel[t,0]], [pos[t,1], bel[t,1]], "-", color="#e8954c", lw=1.2, alpha=0.7)
    ax.plot(pos[t,0], pos[t,1], "o", color="#fff", ms=9)
    ax.plot(bel[t,0], bel[t,1], "o", color="#e8954c", ms=9)
    ax.set_title(f"real GRU belief drifting — step {t}   drift {drift[t]:.2f} cells", color="#e6e9ef", fontsize=11)

print("rendering strip + gif ...")
fst, axs = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#0f1116")
for k, t in enumerate([T//4, T//2, T-1]): frame(axs[k], t)
fst.tight_layout(); fst.savefig("maze_belief_strip.png", dpi=110, facecolor="#0f1116")

figg, axg = plt.subplots(figsize=(5.4, 5.4), facecolor="#0f1116")
anim = animation.FuncAnimation(figg, lambda t: frame(axg, t), frames=T, interval=60)
anim.save("maze_belief.gif", writer=animation.PillowWriter(fps=16))
print(f"saved maze_belief.gif + maze_belief_strip.png  (final drift {drift[-1]:.2f} cells)")
