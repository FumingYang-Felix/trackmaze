"""b14: settle the in-principle question with a STRONG solver. ICM (b11) failed not because the field is
unrecoverable but because it cannot make the non-local segment-flips needed to undo accumulated drift from
sparse (1.6%) wrong chain edges -- it stayed at chain_init. Loopy belief propagation propagates the (dense,
exact) closure evidence globally and escapes that local minimum. If BP recovers the quadrant field to high
accuracy FLAT with size (oracle closures), the field IS recoverable size-invariantly -> the 'grows with size'
verdict was a continuous-approx + weak-solver artifact -> gauge (beyond the global 2-bit) BREAKS.

Vectorized loopy max-sum BP over Z4. Then test oracle vs matcher closures, explore vs dense, vs size.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from heading_attack.b11_bp import drive, chain_init, HALF
from heading_attack.b11b_oracle import oracle_edges
from round3a import wrap


def bp_z4(edges, T, theta=8.0, iters=120, damp=0.3):
    """Loopy MAX-product BP (max-sum in log domain) for Z4 synchronization. edges: (i,j,r,w), constraint
    q_i - q_j = r (mod 4). Anchor node 0 to break the global gauge symmetry. Returns argmax-belief q."""
    src, dst, rho, wt = [], [], [], []
    for (i, j, r, w) in edges:
        src.append(i); dst.append(j); rho.append(r % 4); wt.append(w)      # q_src - q_dst = r
        src.append(j); dst.append(i); rho.append((-r) % 4); wt.append(w)
    src = np.array(src); dst = np.array(dst); rho = np.array(rho); wt = np.array(wt)
    D = len(src)
    rev = np.empty(D, int); rev[0::2] = np.arange(1, D, 2); rev[1::2] = np.arange(0, D, 2)
    M = np.zeros((D, 4))               # max-sum message m_{src->dst}(x_dst)
    U = np.zeros((T, 4)); U[0] = [0.0, -1e6, -1e6, -1e6]   # ANCHOR node 0 = quadrant 0
    ar = np.arange(D)
    for it in range(iters):
        B = U.copy(); np.add.at(B, dst, M)
        excl = B[src] - M[rev]                              # D x 4 : belief at src excluding reverse msg
        excl -= excl.max(1, keepdims=True)
        gmax = excl.max(1)                                  # max over x_src (no boost)
        newM = np.empty((D, 4))
        for xt in range(4):
            matched = (xt + rho) % 4                        # x_src satisfying q_src - q_dst = rho
            newM[:, xt] = np.maximum(excl[ar, matched] + theta * wt, gmax)   # max-product update
        newM -= newM.max(1, keepdims=True)
        M = damp * M + (1 - damp) * newM
    B = U.copy(); np.add.at(B, dst, M)
    return B.argmax(1)


def qacc(q, qtrue):
    return max(np.mean((q - qtrue - off) % 4 == 0) for off in range(4))


def ev(n, rev, mazes=2, theta=4.0):
    accs, errs = [], []
    for mi in range(mazes):
        rec = drive(n, 5000 + mi, rev); T = len(rec)
        edges, qtrue, gh = oracle_edges(rec)
        q = bp_z4(edges, T, theta=theta)
        accs.append(qacc(q, qtrue) * 100)
        # heading error referenced to start
        best_off = max(range(4), key=lambda o: np.mean((q - qtrue - o) % 4 == 0))
        qa = (q - q[0]) % 4
        est = np.array([wrap((gh[t] - gh[0]) + HALF * qa[t]) for t in range(T)])
        tru = np.array([r[3] for r in rec])
        errs.append(np.mean([abs(wrap(est[t] - tru[t])) for t in range(T)]) * 180 / math.pi)
    return np.mean(accs), np.mean(errs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[6, 12, 24, 44])
    ap.add_argument("--mazes", type=int, default=2); ap.add_argument("--rev", type=int, default=1); a = ap.parse_args()
    print(f"b14 loopy-BP Z4 quadrant recovery (ORACLE closures, revisit={a.rev}). q-acc & heading-err vs size.", flush=True)
    print(f"{'n':>4} {'px':>6} | {'q-acc':>7} {'head-err(deg)':>13}", flush=True)
    for n in a.sizes:
        ac, er = ev(n, a.rev, a.mazes)
        print(f"{n:>4} {f'{2*n+1}':>6} | {ac:6.1f}% {er:12.1f}", flush=True)
    print("\nq-acc high & FLAT, err low & flat => field recoverable size-invariantly => gauge breaks (beyond 2-bit).", flush=True)
