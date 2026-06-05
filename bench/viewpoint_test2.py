"""Does CANONICALIZING the omni view into the allo frame (keep phase) give BOTH heading-invariance AND
discriminability -- unlike FFT-magnitude (invariant but lossy)?

Descriptors compared (all from a 360deg / 32-ray omni view at each cell x heading):
  rotinv-mag : |FFT| of geometry            -- invariant, but discards phase (low discriminability).
  canon-geo  : geometry rolled into the allo frame by the (known) heading -- keeps phase, heading-invariant.
  canon+lm   : allo-frame geometry + allo-frame landmark-PRESENCE per direction (no absolute id).
GAP = same-cell-across-heading similarity minus different-cell similarity. Bigger = better place code.
"""
import math, numpy as np
from env import gen_maze
from viewpoint_test import cast_ray

K = 32
def omni_geo(wall, px, py, ang):
    return np.array([cast_ray(wall, px, py, ang + 2*math.pi*i/K) for i in range(K)])

def omni_lm(wall, col, px, py, ang, maxd=24.0):                 # has-landmark per ray (allo direction)
    out = np.zeros(K)
    for i in range(K):
        th = ang + 2*math.pi*i/K; dx, dy = math.cos(th), math.sin(th)
        mX, mY = int(px), int(py); W = wall.shape[0]
        ddx = abs(1/dx) if dx else 1e9; ddy = abs(1/dy) if dy else 1e9
        if dx < 0: sx=-1; sdx=(px-mX)*ddx
        else:      sx=1;  sdx=(mX+1-px)*ddx
        if dy < 0: sy=-1; sdy=(py-mY)*ddy
        else:      sy=1;  sdy=(mY+1-py)*ddy
        for _ in range(300):
            if sdx < sdy: sdx+=ddx; mX+=sx
            else:         sdy+=ddy; mY+=sy
            if not (0<=mY<W and 0<=mX<W): break
            if wall[mY, mX] > 0:
                out[i] = 1.0 if col[mY, mX] > 0 else 0.0; break
    return out

def unit(v): return v / (np.linalg.norm(v) + 1e-9)

def run(n=8, seed=1, n_head=8):
    rng = np.random.default_rng(seed); wall, col, _ = gen_maze(n, 0.15, 1, rng)
    ys, xs = np.where(wall == 0); cells = list(zip(xs.tolist(), ys.tolist())); rng.shuffle(cells); cells = cells[:24]
    heads = [2*math.pi*h/n_head for h in range(n_head)]
    D = {"rotinv-mag": [], "canon-geo": [], "canon+lm": []}; cell_of = []
    for ci, (cx, cy) in enumerate(cells):
        for hi, a in enumerate(heads):
            px, py = cx+0.5, cy+0.5; g = omni_geo(wall, px, py, a); lm = omni_lm(wall, col, px, py, a)
            shift = int(round(a/(2*math.pi)*K))                  # roll into allo frame (heading known): undo the +shift rotation
            gc, lc = np.roll(g, shift), np.roll(lm, shift)
            D["rotinv-mag"].append(unit(np.abs(np.fft.rfft(g))))
            D["canon-geo"].append(unit(gc))
            D["canon+lm"].append(unit(np.concatenate([gc, lc])))
            cell_of.append(ci)
    cell_of = np.array(cell_of)
    print(f"maze {2*n+1}x{2*n+1}, {len(cells)} cells x {n_head} headings, omni K={K}")
    print(f"{'descriptor':>12} | {'same-cell(diff head)':>20} {'diff-cell':>10} {'GAP':>7}")
    for name, L in D.items():
        M = np.stack(L); S = M @ M.T
        same = (cell_of[:, None] == cell_of[None, :]) & ~np.eye(len(M), dtype=bool)
        diff = (cell_of[:, None] != cell_of[None, :])
        print(f"{name:>12} | {S[same].mean():20.3f} {S[diff].mean():10.3f} {S[same].mean()-S[diff].mean():7.3f}")
    print("\ncanon-* should beat rotinv-mag's gap (keeps phase); +lm should add discriminability.")

if __name__ == "__main__":
    run()
