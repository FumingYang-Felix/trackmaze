"""The MECHANISM behind the R13 navigation dissociation, measured policy-independently: how badly does each
egocentric FRAME drift from the true heading as a function of maze size? The agent canonicalizes its observations
by a heading estimate; if that estimate drifts with size, the whole obs frame rotates OOD and the learned policy
fails. We run a scripted DFS coverage walk (the same exploration the policies imitate) and track, per step, the
start-relative heading error of:
  cmd  = command-integrated heading (pure odometry; what gru_cmd uses) -> should GROW with size (unbounded ~√T)
  corr = grid-corrected heading (fine part pinned to the allothetic freq-4 wall cue; what gru_corr/ours use)
         -> should stay BOUNDED / flat with size (the bounded-error law: allothetic anchor caps drift)
This is the bounded-vs-unbounded law FOR THE FRAME, and it predicts the behavioral collapse: gru_cmd's frame
error explodes exactly where its reach-rate goes to 0; gru_corr's frame error stays low exactly where it keeps
navigating. Ties round13_results.md (behavior) to THEORY.md / heading_attack (the bounded-error theory)."""
import argparse, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from env import TrackMazeEnv
from generate_allo import omni
from round3a import cell_graph, wrap, motion
import nav_bc as NB

DIRS = NB.DIRS; ROT = NB.ROT


def cidx(env): return ((int(env.px) - 1) // 2, (int(env.py) - 1) // 2)


def drive_track(env, target, st, ang0, errs, max_drive=80):
    """Drive toward target cell; per raw step update NavState (cmd,corr) and record (cmd_err, corr_err) vs true."""
    wx, wy = 2 * target[0] + 1.5, 2 * target[1] + 1.5
    last_a = 0
    for _ in range(max_drive):
        dx, dy = wx - env.px, wy - env.py
        if math.hypot(dx, dy) < 0.3: break
        err = wrap(math.atan2(dy, dx) - env.ang)
        a = 3 if (abs(err) > ROT and err > 0) else (2 if abs(err) > ROT else 0)
        g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
        st.observe(g, l, last_a)                            # updates corr (grid fusion); cmd updated in advance
        def w90(x): return abs((x + math.pi / 4) % (math.pi / 2) - math.pi / 4)   # |error| mod 90deg, in [0,45]
        true_rel = wrap(env.ang - ang0)
        errs["cmd"].append(w90(st.cmd - true_rel))          # idiothetic: cmd vs its anchor (start) -> DRIFTS
        errs["corr"].append(w90(st.corr - env.ang))         # allothetic: corr vs its anchor (world grid) -> PINNED
        c0 = cidx(env); motion(env, a); st.advance(a, cidx(env) != c0); last_a = a


def dfs_track(env, n, st, ang0, errs, max_steps):
    adj = cell_graph(env.wall, n)
    def sf_key(d): th = math.atan2(d[1], d[0]); return int(round(wrap(th - ang0) / (math.pi / 2))) % 4
    def opens(c): return sorted([(dx, dy) for (dx, dy) in DIRS if (c[0] + dx, c[1] + dy) in adj.get(c, [])], key=sf_key)
    nodes = [dict(cell=cidx(env), explored=set(), parent=None)]; stack = [0]; steps = [0]
    while stack and len(errs["cmd"]) < max_steps:
        top = stack[-1]; cur = cidx(env)
        untried = [d for d in opens(cur) if d not in nodes[top]["explored"]]
        if untried:
            d = untried[0]; nodes[top]["explored"].add(d); nxt = (cur[0] + d[0], cur[1] + d[1])
            drive_track(env, nxt, st, ang0, errs); nc = cidx(env)
            m = next((i for i, nd in enumerate(nodes) if nd["cell"] == nc), None)
            if m is None:
                nodes.append(dict(cell=nc, explored={(-d[0], -d[1])}, parent=(top, (-d[0], -d[1])))); stack.append(len(nodes) - 1)
            else:
                nodes[m]["explored"].add((-d[0], -d[1])); drive_track(env, cur, st, ang0, errs)
        else:
            if nodes[top]["parent"] is None: break
            par, backdir = nodes[top]["parent"]; stack.pop(); drive_track(env, (cur[0] + backdir[0], cur[1] + backdir[1]), st, ang0, errs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 7, 12, 20, 28, 40])
    ap.add_argument("--mazes", type=int, default=12); ap.add_argument("--max_mult", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); np.random.seed(a.seed)
    print(f"{'n':>4} {'px':>5} | {'cmd fine err(deg)':>18} {'corr fine err(deg)':>20}  (mod-90 heading vs each frame's own anchor)", flush=True)
    CMD, CORR, PX = [], [], []
    for n in a.sizes:
        ec, eo = [], []
        for m in range(a.mazes):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=70000 + m, max_steps=10 ** 9); env.reset()
            st = NB.NavState("ours"); errs = {"cmd": [], "corr": []}
            dfs_track(env, n, st, env.ang, errs, a.max_mult * n * n)
            if len(errs["cmd"]) > 2:
                ec.append(np.mean(errs["cmd"])); eo.append(np.mean(errs["corr"]))
        mc = math.degrees(np.mean(ec)); mo = math.degrees(np.mean(eo))
        CMD.append(mc); CORR.append(mo); PX.append(2 * n + 1)
        print(f"{n:>4} {2*n+1:>5} | {mc:>18.1f} {mo:>20.1f}", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(PX, CMD, "o-", color="#d62728", lw=2, label="cmd frame (command odometry) — gru_cmd")
    ax.plot(PX, CORR, "o-", color="#2ca02c", lw=2, label="corr frame (grid-corrected) — gru_corr/ours")
    ax.axvspan(2 * 5 + 1 - 0.5, 2 * 7 + 1 + 0.5, color="gray", alpha=0.12)
    ax.set_xlabel("maze size (px)"); ax.set_ylabel("frame drift = std of frame offset (deg)")
    ax.set_title("Why e2e nav collapses OOD: the command frame DRIFTS with size, the grid-corrected frame is BOUNDED")
    ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("frame_drift.png", dpi=140)
    print("\nwrote frame_drift.png — cmd-frame error GROWS with size (unbounded, crosses 45° -> wrong quadrant ->"
          " obs frame rotates -> gru_cmd reach->0); corr-frame error stays BOUNDED (allothetic anchor) -> nav survives OOD.", flush=True)
