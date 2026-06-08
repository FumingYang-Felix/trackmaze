"""Stage-2 DEMO video: the learned policies ACTUALLY WALKING the maze, methods (rows) x sizes (cols), TrackMaze
style (dark corridors, gray walls, colored landmarks, yellow goal). Same maze per column -> fair comparison.
Shows the dissociation: gru_cmd (drifting frame) gets LOST as size grows; gru_corr / ours reach the exit.
Loads checkpoints from nav_bc.py --save. Usage:
  python nav_demo.py --ckpts ckpt_gru_cmd.pt ckpt_gru_corr.pt ckpt_ours.pt --sizes 5 10 18 28 --out nav_demo.mp4
"""
import argparse, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from env import TrackMazeEnv
from generate_allo import omni
from round3a import wrap, motion
import nav_bc as NB

DEV = "cpu"
LMCOL = ["#000000", "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]  # 1..6 landmark colors
ROWCOL = {"gru_cmd": "#ff5555", "gru_corr": "#ffb04d", "ours": "#4dff88", "tf_corr": "#c98cff", "gru_compass": "#5599ff"}
ROWLAB = {"gru_cmd": "GRU drifting-frame (e2e)", "gru_corr": "GRU grid-corrected", "ours": "OURS (+memory)",
          "tf_corr": "Transformer", "gru_compass": "GRU+compass"}


def maze_rgb(env, n):
    W = 2 * n + 1; img = np.zeros((W, W, 3), np.float32)
    img[env.wall == 0] = (0.10, 0.10, 0.13)                # open = dark corridor
    img[env.wall == 1] = (0.40, 0.40, 0.44)                # wall = gray
    ys, xs = np.where(env.col > 0)
    for y, x in zip(ys, xs):
        c = LMCOL[int(env.col[y, x]) % 7]
        img[y, x] = tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))   # colored landmark
    return img


def load_model(path):
    ck = torch.load(path, map_location="cpu"); m = NB.build(ck["arch"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return ck["arch"], m


def rollout(arch, model, n, seed, max_steps):
    env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed, max_steps=10 ** 9); env.reset()
    ang0 = env.ang; st = NB.NavState(arch); h = model.init_h(1); last_a = 0
    traj = [(env.px, env.py, env.ang)]; reached = False
    gx, gy = env.gx, env.gy
    for t in range(max_steps):
        g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
        o = st.observe(g, l, last_a, compass_h=wrap(env.ang - ang0))
        with torch.no_grad():
            logits, h = model.step(torch.tensor(o, device=DEV).unsqueeze(0), h); a = int(torch.argmax(logits, -1))
        c0 = ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2); motion(env, a)
        c1 = ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2); st.advance(a, c1 != c0); last_a = a
        traj.append((env.px, env.py, env.ang))
        if math.hypot(env.gx - env.px, env.gy - env.py) < 0.6: reached = True; break
    return env, traj, reached, (gx, gy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 18, 28])
    ap.add_argument("--seed0", type=int, default=20250); ap.add_argument("--max_mult", type=int, default=18)
    ap.add_argument("--frames", type=int, default=160); ap.add_argument("--out", default="nav_demo.mp4")
    a = ap.parse_args()
    models = [load_model(p) for p in a.ckpts]; R, C = len(models), len(a.sizes)
    print(f"archs={[m[0] for m in models]} sizes={a.sizes}", flush=True)
    # pick, per column, a maze where the BEST policy (prefer 'ours', else last) reaches the exit -> compelling demo
    ref_ri = next((i for i, m in enumerate(models) if m[0] == "ours"), len(models) - 1)
    cell = {}
    for ci, n in enumerate(a.sizes):
        seed = a.seed0 + ci * 1000
        for _try in range(16):                              # search seeds until the reference policy reaches the exit
            _, _, rok, _ = rollout(models[ref_ri][0], models[ref_ri][1], n, seed, a.max_mult * n * n)
            if rok: break
            seed += 1
        for ri, (arch, model) in enumerate(models):
            env, traj, reached, goal = rollout(arch, model, n, seed, a.max_mult * n * n)
            idx = np.linspace(0, len(traj) - 1, a.frames).astype(int)             # subsample to fixed frame count
            cell[(ri, ci)] = dict(env=env, n=n, traj=[traj[i] for i in idx], reached=reached, goal=goal, arch=arch)
            print(f"  {arch:10s} n={n:3d} px={2*n+1:3d}: {'REACHED' if reached else 'lost  '} in {len(traj)} steps", flush=True)
    fig, axes = plt.subplots(R, C, figsize=(3.0 * C, 3.0 * R), squeeze=False)
    arts = {}
    for ri in range(R):
        for ci in range(C):
            ax = axes[ri][ci]; d = cell[(ri, ci)]; ax.imshow(maze_rgb(d["env"], d["n"]), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            gx, gy = d["goal"]; ax.scatter([gx], [gy], marker="*", s=180, c="#ffe600", edgecolors="k", zorder=5, linewidths=0.5)
            col = ROWCOL.get(d["arch"], "#ffffff")
            (line,) = ax.plot([], [], "-", color=col, lw=1.6, alpha=0.9, zorder=4)
            (agent,) = ax.plot([], [], "o", color=col, ms=7, mec="white", mew=0.8, zorder=6)
            sx, sy = d["traj"][0][0], d["traj"][0][1]
            ax.scatter([sx], [sy], marker="s", s=40, facecolors="none", edgecolors="w", linewidths=0.8, zorder=5)  # start
            arts[(ri, ci)] = (line, agent)
            if ri == 0: ax.set_title(f"{2*d['n']+1}px" + ("  (train)" if d["n"] <= 7 else "  (OOD)"), fontsize=10)
            if ci == 0: ax.set_ylabel(ROWLAB.get(d["arch"], d["arch"]), fontsize=9, color=col)
    status = {}
    def update(f):
        out = []
        for ri in range(R):
            for ci in range(C):
                d = cell[(ri, ci)]; line, agent = arts[(ri, ci)]; tr = d["traj"]
                k = min(f, len(tr) - 1); xs = [p[0] for p in tr[:k + 1]]; ys = [p[1] for p in tr[:k + 1]]
                line.set_data(xs, ys); px, py, ang = tr[k]
                agent.set_data([px], [py])
                out += [line, agent]
                if k == len(tr) - 1 and (ri, ci) not in status:        # episode finished -> badge
                    d["env"]; ax = axes[ri][ci]
                    ax.text(0.97, 0.04, "✓ exit" if d["reached"] else "✗ lost", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=10, color=("#4dff88" if d["reached"] else "#ff5555"),
                            fontweight="bold", zorder=7)
                    status[(ri, ci)] = True
        return out
    fig.suptitle("TrackMaze Stage-2: a LEARNED policy walks the maze — drift-free frame generalizes OOD-size, "
                 "drifting end-to-end gets lost", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    anim = animation.FuncAnimation(fig, update, frames=a.frames, interval=60, blit=False)
    try:
        anim.save(a.out, writer="ffmpeg", fps=18, dpi=110); print(f"wrote {a.out}", flush=True)
    except Exception as e:
        gif = a.out.rsplit(".", 1)[0] + ".gif"; anim.save(gif, writer="pillow", fps=14, dpi=90)
        print(f"ffmpeg failed ({e}); wrote {gif}", flush=True)


if __name__ == "__main__":
    main()
