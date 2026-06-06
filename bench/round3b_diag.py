"""Round 3-B diagnostics: (1) K-sweep -- how small can the compressed pose-graph get before it stops
matching `full`? (2) power-law fit error ~ size^p for full vs compressed (is the residual the sqrt
single-anchor limit?). (3) cmd-heading regime -- confirm heading drift is a SEPARATE dominant bottleneck
that translation memory cannot fix."""
import math, numpy as np
from env import TrackMazeEnv
from round3b import explore_and_log, solve_pose_graph, pick_keep

SIZES = [6, 12, 20, 28, 40]
MAZES = 12


def curve(variant, K, oracle_heading=True):
    out = []
    for n in SIZES:
        errs = []
        for m in range(MAZES):
            env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=3000 + m, max_steps=10 ** 9)
            env.reset()
            nodes, loops = explore_and_log(env, n, oracle_heading=oracle_heading)
            true_xy = [nd[0] for nd in nodes]; est_xy = [nd[1] for nd in nodes]; N = len(nodes)
            keep = pick_keep(N, loops, variant, K)
            errs.append(solve_pose_graph(true_xy, est_xy, loops, keep))
        out.append(np.mean(errs))
    return np.array(out)


def plaw(y):
    p = np.polyfit(np.log(np.array(SIZES)), np.log(y), 1)
    return p[0]   # exponent


if __name__ == "__main__":
    print("=== (1) K-sweep: mean error vs size, oracle heading (lower=better; full N grows ~n^2) ===")
    full = curve("full", 0)
    print(f"  {'variant':>12}  " + " ".join(f"n{n:>3}" for n in SIZES) + f"   exponent_p (sqrt=0.5)")
    print(f"  {'full(O(n^2))':>12}  " + " ".join(f"{v:4.2f}" for v in full) + f"   p={plaw(full):.2f}")
    for K in (10, 20, 40, 80, 160):
        c = curve("loops", K)
        print(f"  {'loops K=%d' % K:>12}  " + " ".join(f"{v:4.2f}" for v in c) + f"   p={plaw(c):.2f}")
    print("\n  => the smallest CONSTANT K whose curve (and exponent) still matches full is the compression floor.")
    print("     exponent p ~ 0.5 == the fundamental single-gauge-anchor sqrt limit (memory-independent).")

    print("\n=== (3) cmd-heading: heading integrated from commands (drifts) -- translation memory can't fix ===")
    fh = curve("full", 0, oracle_heading=False)
    lh = curve("loops", 80, oracle_heading=False)
    print(f"  {'full  (cmd-h)':>12}  " + " ".join(f"{v:5.1f}" for v in fh) + f"   p={plaw(fh):.2f}")
    print(f"  {'loops (cmd-h)':>12}  " + " ".join(f"{v:5.1f}" for v in lh) + f"   p={plaw(lh):.2f}")
    print("  => with heading drift, even full pose-graph blows up (p>>0.5): heading is a SEPARATE bottleneck,")
    print("     fixable by observation-based heading correction (R2 allo-canonical view) / SE(2) loop closure, not memory.")
