"""b11b: fast ORACLE-closure discrete Z4 recovery (no matcher -> isolates whether the FIELD is recoverable).
explore-1x vs dense-3x, ICM solver, flushed output."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from heading_attack.b11_bp import drive, icm, chain_init, HALF
from round3a import wrap


def oracle_edges(rec, npc=8):
    T = len(rec); gh = np.array([r[0] for r in rec])
    qtrue = np.array([int(round((rec[t][3] - rec[t][0] + rec[0][0]) / HALF)) for t in range(T)])
    edges = []
    for t in range(1, T):
        r = int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF)) % 4
        edges.append((t, t - 1, r, 3.0))
    bycell = defaultdict(list)
    for t, rr in enumerate(rec): bycell[rr[2]].append(t)
    for c, ts in bycell.items():
        if len(ts) < 2: continue
        for a in ts[:3]:
            for b in ts[1:1 + npc]:
                if b <= a: continue
                edges.append((b, a, (qtrue[b] - qtrue[a]) % 4, 1.0))
    return edges, qtrue, gh


def chain_acc(rec):
    T = len(rec); gh = np.array([r[0] for r in rec])
    qtrue = np.array([int(round((rec[t][3] - rec[t][0] + rec[0][0]) / HALF)) for t in range(T)])
    ok = sum(1 for t in range(1, T) if int(round((rec[t][1] - (gh[t] - gh[t - 1])) / HALF)) % 4 == (qtrue[t] - qtrue[t - 1]) % 4)
    return ok / max(T - 1, 1)


def ev(n, rev, mazes=2):
    es, cas = [], []
    for mi in range(mazes):
        rec = drive(n, 5000 + mi, rev); T = len(rec)
        edges, qtrue, gh = oracle_edges(rec)
        q = icm(edges, T, chain_init(edges, T))
        est = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
        tru = np.array([r[3] for r in rec])
        es.append(np.mean([abs(wrap(est[t] - tru[t])) for t in range(T)]) * 180 / math.pi)
        cas.append(chain_acc(rec))
    return np.mean(es), np.mean(cas)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44])
    ap.add_argument("--mazes", type=int, default=2); a = ap.parse_args()
    print("b11b ORACLE-closure discrete recovery (ICM). err(deg) vs size. chainAcc = per-step rel-quadrant acc.", flush=True)
    print(f"{'n':>4} {'px':>6} | {'explore-1x':>11} {'dense-3x':>9} {'chainAcc':>9}", flush=True)
    for n in a.sizes:
        e1, ca = ev(n, 1, a.mazes); e3, _ = ev(n, 3, a.mazes)
        print(f"{n:>4} {f'{2*n+1}':>6} | {e1:11.1f} {e3:9.1f} {ca*100:8.1f}%", flush=True)
    print("\ndense-3x flat & low => field recoverable given density. chainAcc<100% => the errors sync must fix.", flush=True)
