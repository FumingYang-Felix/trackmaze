"""Aggregate nav_bc OOD results (navbc_*.txt: 'arch tag seed n reach ratio') over seeds and plot the headline:
reach-rate and steps-vs-teacher ratio vs maze size, one line per arch. The spotlight panel for navigation:
OURS (drift-free frame + spatial memory) stays high/flat across size; gru_cmd (drifting frame) collapses OOD;
gru_corr / tf_corr fall in between -> structure drives OOD-size navigation, end-to-end integration does not."""
import sys, glob, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCH_ORDER = ["gru_cmd", "gru_corr", "tf_corr", "ours", "gru_compass"]
LABEL = {"gru_cmd": "GRU (drifting frame) — e2e", "gru_corr": "GRU + grid-corrected frame",
         "tf_corr": "Transformer + corrected frame", "ours": "OURS (corrected frame + spatial memory)",
         "gru_compass": "GRU + compass (allothetic global cue)"}
COL = {"gru_cmd": "#d62728", "gru_corr": "#ff7f0e", "tf_corr": "#9467bd", "ours": "#2ca02c", "gru_compass": "#1f77b4"}


def load(files):
    D = {}                                                 # arch -> n -> {reach:[], ratio:[]}
    for fn in files:
        for ln in open(fn):
            p = ln.split()
            if len(p) < 5: continue                        # 6 cols = exploration (reach,ratio); 5 = homing (rate)
            arch, n, reach = p[0], int(p[3]), float(p[4])
            ratio = float(p[5]) if len(p) >= 6 else float("nan")
            D.setdefault(arch, {}).setdefault(n, {"reach": [], "ratio": []})
            D[arch][n]["reach"].append(reach)
            if not math.isnan(ratio): D[arch][n]["ratio"].append(ratio)
    return D


def agg(d):
    ns = sorted(d.keys()); m = [np.mean(d[n]["reach"]) for n in ns]
    se = [np.std(d[n]["reach"]) / max(1, math.sqrt(len(d[n]["reach"]))) for n in ns]
    rt = [np.mean(d[n]["ratio"]) if d[n]["ratio"] else np.nan for n in ns]
    return ns, m, se, rt


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("navbc_*.txt"))
    D = load(files)
    if not D: print("no navbc_*.txt found"); sys.exit(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for arch in ARCH_ORDER:
        if arch not in D: continue
        ns, m, se, rt = agg(D[arch])
        xs = [2 * n + 1 for n in ns]
        ax[0].errorbar(xs, m, yerr=se, marker="o", color=COL[arch], label=LABEL[arch], lw=2, capsize=3)
        ax[1].plot(xs, rt, marker="s", color=COL[arch], label=LABEL[arch], lw=2)
    tr = sorted({n for arch in D for n in D[arch]})
    for a_ in ax:
        a_.axvspan(2 * 5 + 1 - 0.5, 2 * 7 + 1 + 0.5, color="gray", alpha=0.12)   # train sizes 5,7
        a_.set_xlabel("maze size (px)")
    ax[0].set_ylabel("reach-rate (exit found)"); ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_title("OOD-size navigation: reach-rate"); ax[0].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("steps / teacher (1=optimal)"); ax[1].set_title("path redundancy")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].text(0.98, 0.96, "gray = train sizes", transform=ax[0].transAxes, ha="right", va="top", fontsize=7, color="gray")
    plt.tight_layout(); plt.savefig("nav_ood_figure.png", dpi=140)
    print("wrote nav_ood_figure.png")
    for arch in ARCH_ORDER:
        if arch not in D: continue
        ns, m, se, rt = agg(D[arch])
        print(f"{arch:9s} reach " + " ".join(f"{2*n+1}px:{mm:.2f}" for n, mm in zip(ns, m)))
