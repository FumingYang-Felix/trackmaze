"""b08: THE decisive test for 'can a learned method + data break the wall'. The fine heading (mod-90) is
drift-free & size-invariant. The ONLY missing piece is the absolute QUADRANT (which of 4), which is gauge-bound
because the maze is (conjectured) exactly 4-fold rotationally symmetric in its LOCAL appearance. But IS it
exactly symmetric? If maze GENERATION (DFS from a corner, landmark placement) leaves ANY global orientation
cue, a learned classifier could read the absolute quadrant from a SINGLE view -> heading fully recoverable,
size-invariantly, by LEARNING (the user's intuition). If accuracy ~= 25% (chance), the 4-fold symmetry is
exact -> the quadrant is information-theoretically unrecoverable from local views -> fundamental.

Train a classifier: (omni geo + lm of one view) -> absolute world quadrant floor(ang/90)%4. Train on small
mazes, eval OOD large. >25% & OOD-stable => symmetry-breaking cue exists & transfers => LEARN heading. ~25%
flat => exact symmetry => fundamental (only a global oriented cue can break it).
"""
import sys, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TrackMazeEnv
from generate_allo import omni, KO
from round3a import cell_graph, dfs_walk, wrap, motion
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HALF = math.pi / 2


def collect(n, n_mazes, seed0, T):
    X, Y = [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=seed0 + m, max_steps=10 ** 9)
        env.reset()
        adj = cell_graph(env.wall, n); walk = dfs_walk(adj, (0, 0)); steps = 0
        for (cx, cy) in walk[1:]:
            if steps >= T: break
            wx, wy = 2 * cx + 1.5, 2 * cy + 1.5
            for _ in range(60):
                if steps >= T: break
                dx, dy = wx - env.px, wy - env.py
                if math.hypot(dx, dy) < 0.3: break
                e = wrap(math.atan2(dy, dx) - env.ang)
                a = 3 if (abs(e) > 0.20 and e > 0) else (2 if abs(e) > 0.20 else 0)
                g, l = omni(env.wall, env.col, env.px, env.py, env.ang)
                X.append(np.concatenate([g, l]).astype(np.float32))
                Y.append(int(math.floor((env.ang % (2 * math.pi)) / HALF)) % 4)
                motion(env, a); steps += 1
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


class Clf(nn.Module):
    def __init__(self, din, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4))
    def forward(self, x): return self.net(x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--eval_sizes", type=int, nargs="+", default=[6, 12, 24, 44, 64])
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    Xtr, Ytr = collect(6, 6, 1000, 240); X2, Y2 = collect(12, 5, 1100, 480)
    Xtr = torch.cat([Xtr, X2]); Ytr = torch.cat([Ytr, Y2]); Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV)
    # class balance check
    print("train class balance:", np.bincount(Ytr.cpu().numpy(), minlength=4))
    m = Clf(Xtr.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(a.epochs):
        idx = torch.randint(0, Xtr.shape[0], (256,), device=DEV)
        loss = F.cross_entropy(m(Xtr[idx]), Ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    tracc = (m(Xtr).argmax(1) == Ytr).float().mean().item()
    print(f"b08 absolute-quadrant-from-one-view. train acc={tracc*100:.1f}% (chance=25%)")
    print(f"{'n':>4} {'grid':>9} | {'OOD acc':>8} (>25% => symmetry-breaking cue => LEARN heading; ~25% => exact 4-fold)")
    for n in a.eval_sizes:
        Xe, Ye = collect(n, 3, 9000, 40 * n); Xe, Ye = Xe.to(DEV), Ye.to(DEV)
        with torch.no_grad(): acc = (m(Xe).argmax(1) == Ye).float().mean().item()
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9} | {acc*100:7.1f}%")
    print("\n>25% & flat OOD => absolute heading LEARNABLE (size-invariant). ~25% => 4-fold symmetry exact => fundamental.")
