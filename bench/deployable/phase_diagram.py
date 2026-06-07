"""SPOTLIGHT main figure (item 2): the PHASE DIAGRAM of global egocentric self-localization. Order parameter
= the estimator-free global-frame heading-std floor (effective resistance to the anchor on the honest VISIT
graph; the BLUE variance no estimator beats). Recoverable iff floor < 45deg (quadrant-recovery threshold).

Two slices:
  (A) loop-density x maze-size  (fixed dense closures): shows the topological order/disorder boundary -- tree
      (loop~0) GROWS with size (disordered, unrecoverable); loopy (high loop) FLAT (ordered, recoverable). The
      critical dimension d=2 boundary.
  (B) loop-density x closure-density (fixed size): shows recoverability also needs enough loop closures
      (coverage x place-recognition) -- the active-SLAM axis.

Saves phase_data.npz + phase_diagram.png. The symmetry axis (the irreducible global 2-bit, collapsed by one
featural cue) is orthogonal and already established (b08/b09) -- annotated on the figure.
"""
import sys, os, math, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from deployable.seqplace import rw_drive   # reuse the loop-covering random walk (records cell + true heading)
HALF = math.pi / 2


def floor_visit(CELL, ANG, ss=0.09, sc=0.05, npc=40, nsamp=80, seed=0):
    """Effective-resistance global-heading-std floor (deg) on the visit graph: chain (var ss^2) + same-cell
    oracle closures (var sc^2). Sampled diagonal of grounded-Laplacian inverse."""
    T = len(CELL)
    if T < 20: return float('nan'), T
    cellk = [c[0] * 100000 + c[1] for c in CELL]
    rows, cols, w = [], [], []; cc = 1.0 / ss ** 2; ccl = 1.0 / sc ** 2
    for t in range(1, T): rows += [t, t - 1]; cols += [t - 1, t]; w += [cc, cc]
    bycell = defaultdict(list)
    for t, c in enumerate(cellk): bycell[c].append(t)
    for c, ts in bycell.items():
        for a in ts[:6]:
            for b in ts[1:1 + npc]:
                if b <= a: continue
                rows += [b, a]; cols += [a, b]; w += [ccl, ccl]
    W = sp.csr_matrix((w, (rows, cols)), shape=(T, T))
    L = (sp.diags(np.array(W.sum(1)).ravel()) - W).tocsc()
    keep = np.arange(1, T); lu = spla.splu(L[keep][:, keep].tocsc())
    rng = np.random.default_rng(seed); samp = rng.choice(np.arange(1, T), min(nsamp, T - 1), replace=False)
    st = []
    for nd in samp:
        e = np.zeros(T - 1); e[nd - 1] = 1.0; y = lu.solve(e); st.append(math.sqrt(max(y[nd - 1], 0)))
    return float(np.mean(st) * 180 / math.pi), T


def cell_floor(n, loop, seed, npc=40):
    CELL_ANG = rw_drive(n, seed, loop)
    G, Larr, CELL, ANG, DR = CELL_ANG
    return floor_visit(CELL, ANG, npc=npc)[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6, 0.9])
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--npcs", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--mazes", type=int, default=2); ap.add_argument("--fixed_n", type=int, default=16)
    a = ap.parse_args()

    print("PHASE DIAGRAM (A): global-heading floor (deg) over loop x size (dense closures). <45 = recoverable.", flush=True)
    print(f"{'loop\\\\n':>8} | " + " ".join(f"{2*nn+1:>6}" for nn in a.sizes), flush=True)
    A = np.zeros((len(a.loops), len(a.sizes)))
    for i, lp in enumerate(a.loops):
        row = []
        for j, nn in enumerate(a.sizes):
            vals = [cell_floor(nn, lp, 5000 + mi) for mi in range(a.mazes)]
            A[i, j] = np.nanmean(vals); row.append(A[i, j])
        print(f"{lp:>8.1f} | " + " ".join(f"{v:6.0f}" for v in row), flush=True)

    print("\nPHASE DIAGRAM (B): floor (deg) over loop x closure-density (npc), fixed n=%d. <45 = recoverable." % a.fixed_n, flush=True)
    print(f"{'loop\\\\npc':>9} | " + " ".join(f"{p:>6}" for p in a.npcs), flush=True)
    B = np.zeros((len(a.loops), len(a.npcs)))
    for i, lp in enumerate(a.loops):
        row = []
        for j, p in enumerate(a.npcs):
            vals = [cell_floor(a.fixed_n, lp, 5000 + mi, npc=p) for mi in range(a.mazes)]
            B[i, j] = np.nanmean(vals); row.append(B[i, j])
        print(f"{lp:>9.1f} | " + " ".join(f"{v:6.0f}" for v in row), flush=True)

    np.savez(os.path.join(os.path.dirname(__file__), "phase_data.npz"),
             A=A, B=B, loops=a.loops, sizes=a.sizes, npcs=a.npcs, fixed_n=a.fixed_n)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k, (M, xt, xl, ttl) in enumerate([
            (A, [2 * s + 1 for s in a.sizes], "maze size (px)", "(A) loop-density × size\n(dense closures)"),
            (B, a.npcs, "loop-closure density (per cell)", "(B) loop-density × coverage\n(fixed size %dpx)" % (2 * a.fixed_n + 1))]):
            im = ax[k].imshow(np.clip(M, 0, 90), aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0, vmax=90)
            ax[k].set_xticks(range(len(xt))); ax[k].set_xticklabels(xt); ax[k].set_xlabel(xl)
            ax[k].set_yticks(range(len(a.loops))); ax[k].set_yticklabels(a.loops); ax[k].set_ylabel("loop density")
            ax[k].set_title(ttl)
            for ii in range(M.shape[0]):
                for jj in range(M.shape[1]):
                    v = M[ii, jj]; ax[k].text(jj, ii, f"{v:.0f}", ha="center", va="center",
                                              color="white" if v > 45 else "black", fontsize=9)
            plt.colorbar(im, ax=ax[k], label="global-heading floor (deg); <45 recoverable")
        fig.suptitle("Phase diagram of OOD egocentric self-localization (order param = global-frame floor; green=recoverable)")
        fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "phase_diagram.png"), dpi=130)
        print("\nsaved phase_diagram.png + phase_data.npz", flush=True)
    except Exception as ex:
        print("plot skipped:", ex, flush=True)
