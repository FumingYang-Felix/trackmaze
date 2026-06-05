"""Sanity harness: does the env + ground truth + metrics reproduce the phenomenon we built the bench for?

We generate trajectories at a small maze and a large maze, run the OPEN-LOOP dead-reckoner (integrate the
agent's own actions, no collision knowledge), and report:
  - the drift curve (error grows over the episode), worse on the large maze (the OOD axis), and
  - the correction curve (error is lower right after an informative landmark, climbs as steps-since grows).
If both show up, the env injects correctable drift and the metrics can see it -> the bench is wired right.
No torch needed; this validates env.py + generate.py + metrics.py.
"""
import numpy as np
from generate import gen_dataset
from metrics import dead_reckon, per_step_error, drift_curve, correction_curve

def run(n, T, n_eps, seed=1):
    ds = gen_dataset(n=n, n_eps=n_eps, T=T, seed=seed)
    # open-loop path integration in the agent's own start-frame: start at origin, heading 0, integrate commands
    est = np.stack([dead_reckon(ds["action"][e], (0.0, 0.0), 0.0) for e in range(n_eps)])
    errs = per_step_error(est, ds["disp"])
    dc = drift_curve(errs, nbins=8)
    cc = correction_curve(errs, ds["reanchor"], max_gap=30)
    print(f"\n=== maze n={n} (grid {2*n+1}x{2*n+1})  LBINS={ds['LBINS']}  "
          f"reanchor_freq={ds['reanchor'].mean():.2f}  final_drift={errs[:,-1].mean():.2f} cells ===")
    print("  DRIFT (err vs step):     " + "  ".join(f"{s}:{v:.2f}" for s, v in dc))
    # show correction at a few gap values
    cc_d = {g: v for g, v, _ in cc}
    pts = [g for g in (0, 1, 2, 5, 10, 20, 30) if g in cc_d]
    print("  CORRECTION (err vs steps-since-reanchor): " + "  ".join(f"{g}:{cc_d[g]:.2f}" for g in pts))
    return errs

if __name__ == "__main__":
    small = run(n=6,  T=200, n_eps=60, seed=1)
    large = run(n=14, T=500, n_eps=60, seed=1)
    print(f"\nSUMMARY: open-loop final drift  small={small[:,-1].mean():.2f}  large={large[:,-1].mean():.2f} cells "
          f"(larger maze => more accumulated drift => OOD gap is real).")
