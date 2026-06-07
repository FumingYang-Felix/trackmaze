"""b03d: integrated heading estimator using the b03c 95% matcher. Decompose true heading as
   true = cmd (command-integrated, drift-free RELATIVE to commands, maintains continuity)
        + drift (= accumulated rot_noise; slowly varying)
We estimate DRIFT (not absolute heading) to avoid b03b's failure (greedy snapping destroyed cmd continuity):
   - drift mod-90  : observed every step from the grid  ((grid - cmd) mod 90)  -> the fine part
   - drift QUADRANT: re-anchored at revisits by the 95% matcher (relative quadrant to a stored visit whose
     heading we trust) -> the discrete part. Aggregated GENTLY (EMA) over closures, never greedily snapped.
h_est = cmd + drift_est. Oracle place-id for this mechanism test (place recognition is a separate axis).

Reports absolute heading error (deg) vs size, vs b02 learned (30->67) and cmd-only. If b03d << that & flat ->
the quadrant is resolved -> heading SOLVED (given place-id) -> Stage-1 unblocked.
"""
import sys, os, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, KO, _grid_heading
from round3a import cell_graph, dfs_walk, wrap, motion
from heading_attack.b03c_learned_matcher import Matcher, collect, make_pairs, ccorr
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HALF = math.pi / 2


def train_matcher(seed=0, epochs=1200):
    rng = np.random.default_rng(seed)
    tr = collect(6, 5, 1000, 240) + collect(12, 4, 1100, 480)
    X, Y = make_pairs(tr, 12000, rng); X, Y = X.to(DEV), Y.to(DEV)
    m = Matcher().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(epochs):
        idx = torch.randint(0, X.shape[0], (256,), device=DEV)
        loss = F.cross_entropy(m(X[idx]), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def matcher_relquad(m, g_now, l_now, g_s, l_s):
    feat = np.concatenate([ccorr(g_s, g_now), ccorr(l_s, l_now)]).astype(np.float32)
    with torch.no_grad():
        return int(m(torch.tensor(feat[None]).to(DEV)).argmax(1).item())  # relative quadrant now vs stored (0..3)


def run_size(n, m, n_mazes, seed0, gain=0.3, fine=0.2):
    errs, errs_cmd = [], []
    for mi in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + mi, max_steps=10 ** 9)
        env.reset(); ang0 = env.ang; cmd = 0.0; h_est = 0.0; grid0 = None
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); store = {}; steps = 0; T = 40 * n
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dxx, dyy = wx - env.px, wy - env.py
                if math.hypot(dxx, dyy) < 0.3: break
                e = wrap(math.atan2(dyy, dxx) - env.ang)
                a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                gh = _grid_heading(g)
                if grid0 is None: grid0 = gh                  # anchor the grid frame to start (resolves ang0)
                hr = wrap(gh - grid0)                          # grid estimate of true heading-from-start (mod 90)
                # FINE (gentle): snap h_est's mod-90 to the grid, WITHOUT changing its quadrant
                kf = round((h_est - hr) / HALF); h_est = h_est + fine * wrap(hr + kf * HALF - h_est)
                cell = (int(env.px), int(env.py))
                # QUADRANT re-anchor via 95% matcher (oracle place-id), GENTLE
                if cell in store:
                    h_s, g_s, l_s, gh_s = store[cell]
                    rq = matcher_relquad(m, g, l, g_s, l_s)    # relative quadrant now vs stored
                    full_rel = wrap(wrap(gh - gh_s) + rq * HALF)   # fine + quadrant -> full relative rotation
                    implied = h_s + full_rel
                    h_est = h_est + gain * wrap(implied - h_est)
                store[cell] = (h_est, g, l, gh)
                tru = wrap(env.ang - ang0)
                errs.append(abs(wrap(h_est - tru))); errs_cmd.append(abs(wrap(cmd - tru)))
                if a == 2: cmd -= 0.20; h_est -= 0.20
                elif a == 3: cmd += 0.20; h_est += 0.20
                motion(env, a); steps += 1
    D = lambda x: np.mean(x) * 180 / math.pi
    return D(errs), D(errs_cmd)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    ap.add_argument("--mazes", type=int, default=4); ap.add_argument("--gain", type=float, default=0.25)
    a = ap.parse_args()
    print("Training b03c matcher..."); m = train_matcher()
    print(f"b03d integrated heading (cmd + grid mod-90 + 95%-matcher GENTLE quadrant re-anchor, oracle place-id)")
    print(f"{'n':>4} {'grid':>8} | {'b03d':>7} {'cmd-only':>8}  (deg; vs b02 learned 30->67)")
    for n in a.sizes:
        h, c = run_size(n, m, a.mazes, 5000, a.gain)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {h:7.1f} {c:8.1f}")
    print("\nb03d << 67 & flat => quadrant resolved => heading SOLVED (with place-id) => Stage-1 unblocked.")
