"""b12: the PHASE DIAGRAM of quadrant recoverability. If b11 shows the discrete field is recoverable given
density, the scientific content is the THRESHOLD: as a function of (closure noise p, revisit density K), is the
recovery error FLAT with size (ordered/recoverable phase) or GROWING (disordered/gauge-bound phase)? This is the
2D-Potts order-disorder transition for our traversal graph. It predicts, for the REAL matcher noise (~5%), the
minimum coverage for size-invariant heading -- the precise sense in which 'gauge breaks with coverage'.

Uses ORACLE closures corrupted i.i.d. with prob p (clean control of edge noise), ICM solver from b11.
Reports recovery error (deg) at small vs large size; FLAT(small~large) => recoverable phase.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from heading_attack.b11_bp import drive, icm, chain_init, HALF
from generate_allo import _grid_heading  # noqa
from round3a import wrap


def build_edges_noisy(rec, p, rng, n_close_per_cell=8):
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
            for b in ts[1:1 + n_close_per_cell]:
                if b <= a: continue
                r = (qtrue[b] - qtrue[a]) % 4                       # oracle relative quadrant
                if rng.random() < p: r = (r + rng.integers(1, 4)) % 4  # corrupt i.i.d.
                edges.append((b, a, r, 1.0))
    return edges, qtrue


def recover_err(n, seed0, n_mazes, revisit, p):
    es = []
    for mi in range(n_mazes):
        rng = np.random.default_rng(1234 + mi)
        rec = drive(n, seed0 + mi, revisit); T = len(rec); gh = np.array([r[0] for r in rec])
        edges, _ = build_edges_noisy(rec, p, rng)
        q = icm(edges, T, chain_init(edges, T))
        est = np.array([wrap((gh[t] - gh[0]) + HALF * ((q[t] - q[0]))) for t in range(T)])
        tru = np.array([r[3] for r in rec])
        es.append(np.mean([abs(wrap(est[t] - tru[t])) for t in range(T)]) * 180 / math.pi)
    return np.mean(es)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=int, default=12); ap.add_argument("--large", type=int, default=64)
    ap.add_argument("--mazes", type=int, default=2)
    a = ap.parse_args()
    print("b12: quadrant-recovery PHASE DIAGRAM. err(deg) at small vs large size; ratio~1 => recoverable (flat).")
    print(f"small=n{a.small} ({2*a.small+1}px)  large=n{a.large} ({2*a.large+1}px)")
    print(f"{'K(revisit)':>10} {'p(noise)':>9} | {'err@small':>10} {'err@large':>10} {'large/small':>11}  phase")
    for K in [1, 3, 5]:
        for p in [0.0, 0.05, 0.10, 0.20, 0.30]:
            es = recover_err(a.small, 5000, a.mazes, K, p)
            el = recover_err(a.large, 5000, a.mazes, K, p)
            ratio = el / max(es, 1e-6)
            phase = "RECOVERABLE" if (ratio < 1.5 and el < 35) else ("marginal" if el < 60 else "GAUGE-BOUND")
            print(f"{K:>10} {p:>9.2f} | {es:>10.1f} {el:>10.1f} {ratio:>11.2f}  {phase}")
    print("\nFlat (ratio~1, low err) up to some p => ordered/recoverable phase. The p where it breaks = the Potts")
    print("threshold; real matcher ~5% sits below it iff K large enough => size-invariant heading with coverage.")
