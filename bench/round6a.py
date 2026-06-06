"""Round 6-A: break the heading vicious cycle with an OBSERVATION-based absolute heading reference from the
maze's Manhattan grid. Axis-aligned walls make the omni ray-distance signal ~90-deg-periodic; the phase of
its frequency-4 Fourier component encodes the grid orientation relative to the agent's heading -> heading
mod 90deg, WITHOUT integrating commands (so it does not drift). If (est - true) mod 90deg has low circular
spread and is size-invariant, this is a per-step absolute heading anchor that bounds the heading random walk
(the R3 dominant bottleneck) -> bounds position drift -> keeps the loop-closure metric gate tight.

Compares against the command-integrated heading (which drifts) over a long trajectory.
"""
import argparse, math, numpy as np
from env import TrackMazeEnv
from generate_allo import omni
from round3a import cell_graph, dfs_walk, wrap, motion

KO = 32


def grid_heading(geo):
    """Estimate heading mod (pi/2) from the freq-4 phase of the 32-ray distance signal."""
    F = np.fft.rfft(geo); ph = np.angle(F[4])      # 4 cycles over 32 samples == 90deg periodicity
    return (-ph / 4.0) % (math.pi / 2)


def circ_std_mod(d, period):
    """Circular std of residuals d that live on a circle of given period."""
    ang = d / period * 2 * math.pi
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    R = math.hypot(C, S)
    return math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12)))) * period / (2 * math.pi)


def run(n, n_mazes=4, seed0=9000, fuse_gain=0.12):
    grid_res, cmd_res, fused_res = [], [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang; cmd = ang0; fused = ang0    # both seeded at the (known) start heading
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0))
        last = None

        def observe():
            nonlocal fused
            g, _ = omni(env.wall, env.col, env.px, env.py, env.ang)
            est90 = grid_heading(g)                               # heading mod 90deg (drift-free, noisy)
            # complementary fuse: snap fused toward the nearest grid-consistent heading (cmd resolves the quadrant)
            k = round((fused - est90) / (math.pi / 2))
            target = est90 + k * (math.pi / 2)
            fused += fuse_gain * wrap(target - fused)
            tru = env.ang
            grid_res.append(wrap_to(est90 - tru % (math.pi / 2), math.pi / 2))
            cmd_res.append(wrap(cmd - tru))
            fused_res.append(wrap(fused - tru))

        observe(); last = (int(env.px), int(env.py))
        for (cx, cy) in walk[1:]:
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(40):
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                err = wrap(math.atan2(dy, dx) - env.ang)
                if abs(err) > 0.20:
                    if err > 0: cmd += 0.20; fused += 0.20; a = 3
                    else:       cmd -= 0.20; fused -= 0.20; a = 2
                else:
                    a = 0
                motion(env, a)
                cell = (int(env.px), int(env.py))
                if cell != last:
                    observe(); last = cell
    return np.array(grid_res), np.array(cmd_res), np.array(fused_res)


def wrap_to(x, period):
    return (x + period / 2) % period - period / 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 20, 28, 40])
    a = ap.parse_args()
    print("Heading error (full 360deg) vs size: CMD = command-integrated (drifts); FUSED = CMD + grid-obs anchor.")
    print("GRID_std = spread of the raw mod-90 observation (size-invariant anchor). RMS in degrees.")
    print(f"{'n':>4} {'grid':>8} | {'GRID_std°':>9} | {'CMD_rms°':>9} | {'FUSED_rms°':>11}")
    D = lambda v: v * 180 / math.pi
    rows = {}
    for n in a.sizes:
        gr, cm, fu = run(n)
        gstd = D(circ_std_mod(gr, math.pi / 2))
        crms = D(math.sqrt(np.mean(cm ** 2))); frms = D(math.sqrt(np.mean(fu ** 2)))
        rows[n] = (crms, frms)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {gstd:9.1f} | {crms:9.1f} | {frms:11.1f}")
    ns = a.sizes
    print(f"\nCMD_rms grows {rows[ns[0]][0]:.0f}->{rows[ns[-1]][0]:.0f}deg (the heading random walk).")
    print(f"FUSED_rms {rows[ns[0]][1]:.0f}->{rows[ns[-1]][1]:.0f}deg: if flat & small, heading is BOUNDED at any size")
    print("=> the observation anchor breaks the heading horn of the loop-closure vicious cycle.")
