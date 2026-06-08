"""Plot the deployable-SLAM scaling result AFTER the per-cell closure-density fix, vs the OLD (broken) numbers,
to show the STORY-S6 >=65px wall is closed. Reads the sweep .out (the AGGREGATE table). Usage:
  python deployable/scaling_figure.py scaling_npc40_<job>.out
"""
import sys, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# OLD deployable heading error (fixed-ncl closures), from scaling_result.txt / the S6 narrative: flat to 49px,
# then ERRATIC at >=65px (50.6 deg outlier at 65px, 14.2 at 129px) = the closure-density collapse / under-determination.
OLD_PX = [17, 33, 49, 65, 89, 129]
OLD_GI = [1.8, 3.1, 3.6, 50.6, 6.3, 14.2]


def parse(path):
    px, gi, qa = [], [], []
    inagg = False
    for ln in open(path, encoding="utf-8", errors="ignore"):
        if "AGGREGATE" in ln: inagg = True; continue
        if not inagg: continue
        m = re.match(r"\s*(\d+)\s+(\d+)\s*\|\s*([\d.]+)\s+([\d.\-]+)\s+([\d.]+)\s+(\d+)%\s+(\d+)%", ln)
        if m:
            px.append(int(m.group(2))); gi.append(float(m.group(5))); qa.append(int(m.group(6)))
    return px, gi, qa


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scaling_npc40.out"
    px, gi, qa = parse(path)
    if not px: print(f"no AGGREGATE table found in {path}"); sys.exit(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(OLD_PX, OLD_GI, "s--", color="#d62728", lw=1.6, alpha=0.7, label="before (fixed closures) — S6 wall")
    ax[0].plot(px, gi, "o-", color="#2ca02c", lw=2.2, label="after (per-cell density fix)")
    ax[0].axhline(45, color="gray", ls=":", lw=1); ax[0].text(px[-1], 47, "45° quadrant threshold", ha="right", fontsize=8, color="gray")
    ax[0].axvspan(64, 132, color="orange", alpha=0.08); ax[0].text(98, ax[0].get_ylim()[1]*0.5, "S6 wall\n(≥65px)", ha="center", fontsize=8, color="darkorange")
    ax[0].set_xlabel("maze size (px)"); ax[0].set_ylabel("gauge-invariant heading error (deg)")
    ax[0].set_title("Deployable global heading scales past the S6 wall"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, max(55, max(gi) * 1.2))
    ax[1].plot(px, qa, "o-", color="#1f77b4", lw=2.2); ax[1].axhline(90, color="gray", ls=":", lw=1)
    ax[1].text(px[0], 91, "90% (recoverable)", fontsize=8, color="gray")
    ax[1].set_xlabel("maze size (px)"); ax[1].set_ylabel("quadrant-field accuracy (%)"); ax[1].set_ylim(0, 105)
    ax[1].set_title("Z₄ quadrant field recovered at every size")
    plt.tight_layout(); plt.savefig("deployable/scaling_fixed.png", dpi=140)
    print("wrote deployable/scaling_fixed.png")
    for p, g, q in zip(px, gi, qa): print(f"  {p}px: gauge-inv={g:.1f}deg  q-acc={q}%")
