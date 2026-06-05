"""Training-free test of the viewpoint/ray critique: is the place signature heading-INVARIANT?

At a fixed cell the agent can face any of 360 degrees, so a place code must recognize "same place" across
headings. We compare three descriptors of the local view, measuring same-cell-across-heading similarity vs
different-cell similarity (the GAP = how well it identifies a place independent of viewpoint):
  1. forward-69 (current): 16 rays over a 69 deg forward cone   -> heading-DEPENDENT (the suspected bug).
  2. omni-360 raw:         32 rays around the full circle        -> rotates (cyclic shift) with heading.
  3. omni-360 rot-invariant: |FFT| of the 32-ray vector          -> rotation/heading-INVARIANT by construction.
Uses wall-distance geometry (the "surroundings", not just landmarks), exactly the user's point.
"""
import math, numpy as np
from env import gen_maze

def cast_ray(wall, px, py, theta, maxd=24.0):
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
        if not (0 <= mY < W and 0 <= mX < W): return 1.0
        if wall[mY, mX] > 0:
            pd = (sdx-ddx) if side==0 else (sdy-ddy); return min(pd, maxd)/maxd
    return 1.0

def forward69(wall, px, py, ang, K=16, fov=math.pi/2.6):
    return np.array([cast_ray(wall, px, py, ang + fov*((i+0.5)/K - 0.5)) for i in range(K)])

def omni(wall, px, py, ang, K=32):
    return np.array([cast_ray(wall, px, py, ang + 2*math.pi*i/K) for i in range(K)])

def rotinv(vec):                       # |rFFT| of the omni ray vector -> invariant to cyclic shift (heading)
    return np.abs(np.fft.rfft(vec))

def unit(v): return v / (np.linalg.norm(v) + 1e-9)

def run(n=8, seed=1, n_head=8):
    rng = np.random.default_rng(seed); wall, col, _ = gen_maze(n, 0.15, 1, rng)
    ys, xs = np.where(wall == 0); cells = list(zip(xs.tolist(), ys.tolist()))
    rng.shuffle(cells); cells = cells[:24]
    heads = [2*math.pi*h/n_head for h in range(n_head)]
    descs = {"fwd69": [], "omni": [], "rotinv": []}; cell_of = []
    for ci, (cx, cy) in enumerate(cells):
        for a in heads:
            px, py = cx+0.5, cy+0.5
            descs["fwd69"].append(unit(forward69(wall, px, py, a)))
            o = omni(wall, px, py, a)
            descs["omni"].append(unit(o)); descs["rotinv"].append(unit(rotinv(o))); cell_of.append(ci)
    cell_of = np.array(cell_of)
    print(f"maze {2*n+1}x{2*n+1}, {len(cells)} cells x {n_head} headings")
    print(f"{'descriptor':>10} | {'same-cell (diff head)':>22} {'diff-cell':>10} {'GAP':>7}")
    for name, D in descs.items():
        D = np.stack(D); S = D @ D.T
        same = (cell_of[:, None] == cell_of[None, :]); np.fill_diagonal(same, False)
        diff = ~same; np.fill_diagonal(diff, True); diff = ~np.eye(len(D), dtype=bool) & ~same
        ss, dd = S[same].mean(), S[diff].mean()
        print(f"{name:>10} | {ss:22.3f} {dd:10.3f} {ss-dd:7.3f}")
    print("\nbig GAP = recognizes the place ACROSS headings. fwd69 should be poor; rotinv should win.")

if __name__ == "__main__":
    run()
