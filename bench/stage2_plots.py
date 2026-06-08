"""Stage-2 overview: one panel per sub-part.
  2a OPEN-arena homing (itwm pi_home): RNN vs FF final distance-to-start vs outbound length, per cue-rate p.
     -> RNN << FF up to ~4x train length, then COMPOUNDS OOD (metric route fails OOD).
  2b MAZE goal-nav / find-exit (nav_bc, the HEADLINE): reach-rate vs maze size, per arch.
     -> drifting-frame e2e collapses; drift-free grid-corrected frame generalizes; compass sustains extreme OOD.
  2c MAZE homing / return-to-start (nav_home --pos): home-rate vs maze size, per arch.
     -> the MEMORY lever shows here (ours > gru_corr > gru_cmd); low absolute + noisy OOD.
"""
import glob, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ARCH_ORDER = ["gru_cmd", "gru_corr", "tf_corr", "ours", "gru_compass"]
LABEL = {"gru_cmd": "GRU drifting-frame (e2e)", "gru_corr": "GRU grid-corrected frame",
         "tf_corr": "Transformer (corrected)", "ours": "OURS (corrected + memory)",
         "gru_compass": "GRU + compass (global cue)"}
COL = {"gru_cmd": "#d62728", "gru_corr": "#ff7f0e", "tf_corr": "#9467bd", "ours": "#2ca02c", "gru_compass": "#1f77b4"}


def load_by_arch(files, col_idx=4):
    D = {}
    for fn in files:
        for ln in open(fn):
            p = ln.split()
            if len(p) <= col_idx: continue
            D.setdefault(p[0], {}).setdefault(int(p[3]), []).append(float(p[col_idx]))
    return D


def agg(d):
    ns = sorted(d); m = [np.mean(d[n]) for n in ns]
    se = [np.std(d[n]) / max(1, math.sqrt(len(d[n]))) for n in ns]
    return [2 * n + 1 for n in ns], m, se


fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

# ---- 2a OPEN homing (hardcoded from job 19961674) ----
T = [30, 60, 120, 240, 480]
RNN = {0.2: [0.12, 0.12, 0.53, 5.61, 19.73], 0.1: [0.16, 0.18, 0.88, 6.83, 16.09], 0.05: [0.69, 0.68, 1.52, 6.48, 20.79]}
FF = {0.2: [1.30, 2.52, 7.80, 11.82, 15.34], 0.1: [3.66, 6.48, 12.79, 20.47, 30.17], 0.05: [6.33, 9.95, 16.28, 24.99, 35.52]}
pcol = {0.2: "#2ca02c", 0.1: "#ff7f0e", 0.05: "#d62728"}
for p in [0.2, 0.1, 0.05]:
    ax[0].plot(T, RNN[p], "o-", color=pcol[p], lw=2, label=f"RNN  p={p}")
    ax[0].plot(T, FF[p], "s--", color=pcol[p], lw=1.3, alpha=0.7, label=f"FF   p={p}")
ax[0].axvspan(55, 65, color="gray", alpha=0.12); ax[0].text(60, ax[0].get_ylim()[1]*0.95, "train", ha="center", fontsize=8, color="gray")
ax[0].set_xlabel("outbound length (steps)"); ax[0].set_ylabel("final distance to start (cells; lower=better)")
ax[0].set_title("2a · OPEN-arena homing\nRNN<<FF, but RNN COMPOUNDS OOD"); ax[0].legend(fontsize=6, ncol=2)

# ---- 2b MAZE goal-nav (headline) ----
Db = load_by_arch(sorted(glob.glob("navbc_strong_*.txt")), col_idx=4)
for a_ in ARCH_ORDER:
    if a_ not in Db: continue
    xs, m, se = agg(Db[a_]); ax[1].errorbar(xs, m, yerr=se, marker="o", color=COL[a_], lw=2, capsize=3, label=LABEL[a_])
ax[1].axvspan(10.5, 15.5, color="gray", alpha=0.12); ax[1].set_ylim(-0.03, 1.0)
ax[1].set_xlabel("maze size (px)"); ax[1].set_ylabel("reach-rate (exit found)")
ax[1].set_title("2b · MAZE find-exit  (HEADLINE)\ndrift-free frame generalizes; e2e collapses"); ax[1].legend(fontsize=7, loc="upper right")

# ---- 2c MAZE homing ----
Dc = load_by_arch(sorted(glob.glob("navhome_*.txt")), col_idx=4)
for a_ in ARCH_ORDER:
    if a_ not in Dc: continue
    xs, m, se = agg(Dc[a_]); ax[2].errorbar(xs, m, yerr=se, marker="o", color=COL[a_], lw=2, capsize=3, label=LABEL[a_])
ax[2].axvspan(10.5, 15.5, color="gray", alpha=0.12); ax[2].set_ylim(-0.03, 0.8)
ax[2].set_xlabel("maze size (px)"); ax[2].set_ylabel("home-rate (returned to start)")
ax[2].set_title("2c · MAZE homing (cognitive map)\nmemory lever: ours > gru_corr > gru_cmd"); ax[2].legend(fontsize=7, loc="upper right")

plt.tight_layout(); plt.savefig("stage2_overview.png", dpi=140)
print("wrote stage2_overview.png")
for tag, D in [("2b find-exit", Db), ("2c homing", Dc)]:
    for a_ in ARCH_ORDER:
        if a_ in D:
            xs, m, se = agg(D[a_]); print(f"{tag:12s} {a_:12s} " + " ".join(f"{x}:{v:.2f}" for x, v in zip(xs, m)))
