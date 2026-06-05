"""Experiment 1+2 (training-free): is the OOD failure a CLIFF, and is the mechanism data-association?

We give the agent an ORACLE re-anchor: whenever it is next to a landmark it knows its offset from
*a* landmark of that id (perfect perception) -- but it must still ASSOCIATE that observation to the
RIGHT instance by picking the same-id landmark cell nearest to its current (drifted) estimate. If drift
is small it matches the true one -> estimate snaps back (bounded error). If drift exceeds ~half the
spacing to another same-id landmark (more likely as ambiguity rises / maze grows), it matches the WRONG
one -> the estimate jumps to the wrong place -> catastrophic. The open-loop integrator (no re-anchor)
is the reference.

Sweeping ambiguity (vocabulary size) and maze size, we read off:
  - does error-vs-ambiguity have a KNEE (the data-association threshold)?  -> tests C2
  - does the oracle re-anchor keep error BOUNDED vs maze size while open-loop grows? -> tests C3
No learned model, no torch: this isolates the geometry of the threshold from any tracker's quality.
"""
import math
import numpy as np
from env import TrackMazeEnv
from generate import explore_policy

MV, ROT = 0.22, 0.20

def landmark_cells_by_id(col):
    d = {}
    ys, xs = np.where(col > 0)
    for y, x in zip(ys.tolist(), xs.tolist()): d.setdefault(int(col[y, x] - 1), []).append((y, x))
    return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}

def integ(x, y, ang, a):
    if a == 2: ang -= ROT
    elif a == 3: ang += ROT
    else:
        f = MV if a == 0 else -MV; x += math.cos(ang) * f; y += math.sin(ang) * f
    return x, y, ang

def run_episode(n, ambiguity, T, seed, lm_density=0.15):
    env = TrackMazeEnv(n=n, ambiguity=ambiguity, lm_density=lm_density, seed=seed, max_steps=T + 2)
    obs = env.reset(); lmc = landmark_cells_by_id(env.col); rng = np.random.default_rng(seed)
    ex, ey, ea = env.px, env.py, env.ang        # open-loop estimate (integrate exact commands, no noise/collision)
    rx, ry, ra = ex, ey, ea                     # re-anchor estimate (same, but snaps at associated landmarks)
    ol, rc = [], []
    for t in range(T):
        a = explore_policy(obs, rng)
        ex, ey, ea = integ(ex, ey, ea, a); rx, ry, ra = integ(rx, ry, ra, a)
        obs, _, _, _ = env.step(a)
        tx, ty = env.px, env.py; cx, cy = int(tx), int(ty)
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):     # landmarks adjacent to the true cell are "in view"
            wx, wy = cx+dx, cy+dy
            if 0<=wy<env.col.shape[0] and 0<=wx<env.col.shape[1] and env.col[wy, wx] > 0:
                lid = int(env.col[wy, wx]-1); cands = lmc[lid]
                j = int(np.argmin((cands[:,1]-rx)**2 + (cands[:,0]-ry)**2))   # ASSOCIATE: nearest same-id to estimate
                my, mx = cands[j]
                rx = (mx+0.5) + (tx-(wx+0.5)); ry = (my+0.5) + (ty-(wy+0.5))  # pin to matched (true if correct, wrong if aliased)
        ol.append(math.hypot(ex-tx, ey-ty)); rc.append(math.hypot(rx-tx, ry-ty))
    return float(np.mean(ol[-T//4:])), float(np.mean(rc[-T//4:]))   # steady-state (last quarter) error

def sweep(label, configs, n_eps=40):
    print(f"\n=== {label} (steady-state error, cells: open-loop -> oracle-reanchor) ===")
    for cfg in configs:
        ols, rcs = [], []
        for e in range(n_eps):
            o, r = run_episode(seed=1000+e, **cfg)
            ols.append(o); rcs.append(r)
        tag = " ".join(f"{k}={v}" for k, v in cfg.items() if k != "T")
        print(f"  {tag:28s}  open {np.mean(ols):.2f}   reanchor {np.mean(rcs):.2f}")

if __name__ == "__main__":
    # (C2) sweep ambiguity on a large maze: does the re-anchor blow up past a knee?
    sweep("ambiguity sweep (n=12, T=360)",
          [dict(n=12, ambiguity=a, T=360) for a in (0,1,2,3)])
    # (C3) sweep size at unique landmarks: does oracle re-anchor stay flat while open-loop grows?
    sweep("size sweep (ambiguity=0/unique, T=300)",
          [dict(n=n, ambiguity=0, T=300) for n in (4,6,9,12,16)])
    # (C3) same but with aliasing (ambiguity=2): does the bound break at large size?
    sweep("size sweep (ambiguity=2/aliased, T=300)",
          [dict(n=n, ambiguity=2, T=300) for n in (4,6,9,12,16)])
