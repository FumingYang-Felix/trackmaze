"""Comprehensive scaling/robustness sweep of the deployable SLAM (designed for a big CPU node, e.g. FASRC).
Parallelizes over (size, maze) with multiprocessing. Reports, per size: online drift, deploy-SLAM heading error
both referenced-to-start AND gauge-invariant (best of the 4 global rotations -- the irreducible 2-bit), plus
q-accuracy and closure accuracy. BP iterations scale with size. Run:
    python deployable/sweep_scaling.py --sizes 8 16 24 32 44 64 89 129 --mazes 5 --workers 32
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from round3a import wrap
import deployable.deploy_slam as ds
from heading_attack.b14_bp_z4 import bp_z4
HALF = math.pi / 2


def one(args):
    n, mi, loop, K, iters, ncl = args
    GH, TURN, CELL, ANG, BR = ds.drive(n, 7000 + mi, loop, K=K)
    T = len(GH)
    cr = np.array([0] + [int(round((TURN[t] - (GH[t] - GH[t - 1])) / HALF)) % 4 for t in range(1, T)])
    qtrue = np.array([int(round((ANG[t] - (GH[t] - GH[0])) / HALF)) for t in range(T)])
    byc = defaultdict(list)
    for t, c in enumerate(CELL): byc[c].append(t)
    fv = {c: ts[0] for c, ts in byc.items()}
    rng = np.random.default_rng(1); qs = rng.choice(T, min(ncl, T), replace=False); cl = []
    for t in qs:
        j = fv[CELL[t]]
        if j >= t: continue
        rot = ds.beacon_rot(BR[t], BR[j])
        if rot is None: continue
        cl.append((int(t), int(j), int(round((rot - (GH[t] - GH[j])) / HALF)) % 4))
    cacc = np.mean([(qtrue[i] - qtrue[j]) % 4 == rel for (i, j, rel) in cl]) if cl else 0.0
    edges = [(t, t - 1, int(cr[t]), 3.0) for t in range(1, T)] + [(i, j, rel, 1.5) for (i, j, rel) in cl]
    q = bp_z4(edges, T, theta=6.0, iters=iters, damp=0.2)
    qacc = max(np.mean((q - qtrue - o) % 4 == 0) for o in range(4))
    base = np.array([GH[t] - GH[0] for t in range(T)])
    e_ref = np.mean([abs(wrap(base[t] + HALF * (q[t] - q[0]) - ANG[t])) for t in range(T)]) * 180 / math.pi
    e_gi = min(np.mean([abs(wrap(base[t] + HALF * (q[t] + g) - ANG[t])) for t in range(T)]) for g in range(4)) * 180 / math.pi
    on = ds.online_err(GH, TURN, ANG)
    return n, on, e_ref, e_gi, qacc * 100, cacc * 100, T


if __name__ == "__main__":
    import argparse, multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32, 44, 64, 89, 129])
    ap.add_argument("--mazes", type=int, default=5); ap.add_argument("--loop", type=float, default=0.9)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--ncl", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    iters_of = lambda n: int(400 + 60 * n)     # BP iterations scale ~ with maze size
    tasks = [(n, mi, a.loop, a.K, iters_of(n), a.ncl) for n in a.sizes for mi in range(a.mazes)]
    print(f"sweep: {len(tasks)} runs on {a.workers} workers. loop={a.loop} K={a.K}", flush=True)
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, tasks)
    by = defaultdict(list)
    for r in res: by[r[0]].append(r)
    print(f"{'n':>4} {'px':>5} | {'online':>7} {'ref-to-start':>12} {'gauge-inv':>9} {'q-acc':>6} {'closAcc':>7} {'T':>7}", flush=True)
    for n in sorted(by):
        rs = by[n]; arr = np.array([[x[1], x[2], x[3], x[4], x[5], x[6]] for x in rs]).mean(0)
        print(f"{n:>4} {2*n+1:>5} | {arr[0]:7.1f} {arr[1]:12.1f} {arr[2]:9.1f} {arr[3]:6.0f}% {arr[4]:6.0f}% {int(arr[5]):7d}", flush=True)
    print("\ngauge-inv flat&low across all sizes => deployable method scales (the irreducible 2-bit aside).", flush=True)
