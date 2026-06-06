"""Stage-1 data: omnidirectional view canonicalized into the ALLO frame (the validated place signature).

At each step the agent senses a 360deg / 32-ray omni view (geometry + landmark-presence per ray) at its
TRUE heading, then rolls it into the allocentric (start) frame using a heading estimate:
  canon='true' -> roll by the true heading-from-start (oracle descriptor; tests if a perfect allo code
                  makes matching/correction finally work).
  canon='cmd'  -> roll by the COMMAND-integrated heading (realistic; drifts with rot_noise).
Also records actions, GT position/displacement, true cell, re-anchor flags.
"""
import math, numpy as np
from env import TrackMazeEnv
from generate import explore_policy, egocentric_disp

KO = 32
def cast(wall, col, px, py, theta, maxd=24.0):
    dx, dy = math.cos(theta), math.sin(theta); W = wall.shape[0]
    mX, mY = int(px), int(py)
    ddx = abs(1/dx) if dx else 1e9; ddy = abs(1/dy) if dy else 1e9
    if dx < 0: sx=-1; sdx=(px-mX)*ddx
    else:      sx=1;  sdx=(mX+1-px)*ddx
    if dy < 0: sy=-1; sdy=(py-mY)*ddy
    else:      sy=1;  sdy=(mY+1-py)*ddy
    side = 0
    for _ in range(300):
        if sdx < sdy: sdx+=ddx; mX+=sx; side=0
        else:         sdy+=ddy; mY+=sy; side=1
        if not (0 <= mY < W and 0 <= mX < W): return 1.0, 0.0
        if wall[mY, mX] > 0:
            pd = (sdx-ddx) if side==0 else (sdy-ddy)
            return min(pd, maxd)/maxd, (1.0 if col[mY, mX] > 0 else 0.0)
    return 1.0, 0.0

def omni(wall, col, px, py, ang):
    g = np.empty(KO, np.float32); l = np.empty(KO, np.float32)
    for i in range(KO):
        g[i], l[i] = cast(wall, col, px, py, ang + 2*math.pi*i/KO)
    return g, l

def _grid_heading(g):
    """R6-A: heading mod (pi/2) from the freq-4 phase of the 32-ray distance signal (drift-free, ~20deg noise)."""
    F = np.fft.rfft(g); return (-np.angle(F[4]) / 4.0) % (math.pi / 2)


def _wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi


def gen_allo(n, n_eps, T, ambiguity=1, seed=0, canon="true", action_noise=0.03, rot_noise=0.09, loop=0.0, fuse_gain=0.3):
    rng = np.random.default_rng(seed); mv, rot = 0.22, 0.20
    CG, CL, ACT, POS, DISP, CELL, RA, STEP = [], [], [], [], [], [], [], []
    for e in range(n_eps):
        env = TrackMazeEnv(n=n, ambiguity=ambiguity, seed=seed*100000+e, max_steps=T+2,
                           action_noise=action_noise, rot_noise=rot_noise, loop=loop)
        obs = env.reset(); ang0 = env.ang; cmd = 0.0; corr = 0.0   # cmd-integrated & grid-CORRECTED headings
        cg, cl, ac, po, ra, st = [], [], [], [np.array([env.px, env.py], np.float32)], [], []
        for t in range(T):
            a = explore_policy(obs, rng)
            g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
            if canon == "corrected":                           # fuse cmd heading with grid-observation (R6-A)
                gobs_rel = (_grid_heading(g) - (ang0 % (math.pi/2))) % (math.pi/2)  # start-relative heading mod 90
                k = round((corr - gobs_rel) / (math.pi/2)); target = gobs_rel + k*(math.pi/2)
                corr += fuse_gain * _wrap(target - corr)
                h = corr
            else:
                h = (env.ang - ang0) if canon == "true" else cmd
            sh = int(round(h/(2*math.pi)*KO))                 # roll into allo (start) frame
            cg.append(np.roll(g, sh)); cl.append(np.roll(l, sh)); ac.append(a)
            f = mv if a == 0 else (-mv if a == 1 else 0.0)    # COMMANDED step in the canonicalization frame (odometry)
            hf = corr if canon == "corrected" else cmd
            st.append(np.array([f*math.cos(hf), f*math.sin(hf)], np.float32))
            if a == 2: cmd -= rot; corr -= rot
            elif a == 3: cmd += rot; corr += rot
            obs, _, _, gt = env.step(a); po.append(gt["pos"]); ra.append(gt["reanchor"])
        pos = np.stack(po[:T]); cells = np.floor(pos).astype(np.int64)
        CG.append(np.stack(cg)); CL.append(np.stack(cl)); ACT.append(np.array(ac, np.int64))
        POS.append(pos); DISP.append(egocentric_disp(pos, ang0)); STEP.append(np.stack(st))
        CELL.append(cells[:, 1]*1000 + cells[:, 0]); RA.append(np.array(ra[:T], bool))
    return dict(canon_geo=np.stack(CG), canon_lm=np.stack(CL), action=np.stack(ACT), cmd_step=np.stack(STEP),
                pos=np.stack(POS), disp=np.stack(DISP), cell=np.stack(CELL), reanchor=np.stack(RA), n=n, KO=KO)

if __name__ == "__main__":
    ds = gen_allo(n=6, n_eps=3, T=80, canon="true")
    for k, v in ds.items(): print(k, getattr(v, "shape", v))
