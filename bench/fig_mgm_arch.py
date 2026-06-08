"""Architecture data-flow diagram for MGM (motion-gated memory tracker). Saves fig_mgm_arch.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(inp="#E8E8E8", enc="#CFE2F3", gru="#D9EAD3", mem="#FCE5CD",
         lc="#F4CCCC", out="#D9D2E9", star="#E69138")
EC = "#555555"

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, text, fc, fs=10, bold=False, ec=EC, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)


def arrow(x1, y1, x2, y2, style="-|>", color=EC, lw=1.6, ls="-", rad=0.0, txt=None, tx=0, ty=0, tc=EC):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))
    if txt:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, txt, fontsize=8.5, color=tc,
                ha="center", va="center", style="italic")


# ---- title
ax.text(8, 9.6, "MGM — motion-gated memory tracker  (one online timestep $t$)",
        ha="center", fontsize=14, fontweight="bold")

# ---- inputs
box(0.3, 6.6, 2.3, 1.5, "omni view\ngeo[32] · lm[32]\n(allo-frame)", C["inp"], 9.5)
box(0.3, 2.3, 2.3, 1.1, "motion\ncmd_step[2]", C["inp"], 9.5)

# ---- encoder
box(3.2, 6.5, 2.6, 1.7, "Obs Encoder\ncirc-conv + rot-pool\n(or MLP, mgm_norot)\n→ place code  $p$ (64)",
    C["enc"], 9)

# ---- GRU
box(6.5, 5.3, 2.5, 1.7, "GRUCell\n$h$ (256)\nlocal integration", C["gru"], 9.5)
# recurrence
arrow(7.75, 7.0, 7.75, 7.7, rad=0); arrow(7.2, 7.7, 7.2, 7.0, rad=0)
ax.add_patch(FancyArrowPatch((7.2, 7.7), (7.75, 7.7), arrowstyle="-", color=EC, lw=1.6))
ax.text(7.47, 7.95, "$h_{t-1}$", fontsize=8.5, ha="center", color=EC)

# ---- mu update
box(9.7, 5.3, 3.1, 1.7,
    "Position update\n$\\mu = \\mu_{t-1} + \\mathrm{motion} + \\mathrm{to\\mu}(h)$\n(dead-reckon + residual)",
    C["gru"], 9)

# ---- loop-closure gate
box(9.7, 2.9, 3.1, 1.6,
    "Loop-closure gate  (×$n_{corr}$)\n$c=\\sigma(\\mathrm{gate}[h,\\,read_p,\\,read_\\mu-\\mu])$\n"
    "$\\mu \\leftarrow \\mu + c\\,(read_\\mu - \\mu)$",
    C["lc"], 8.5)

# ---- memory module (the star)
box(3.0, 0.3, 6.2, 2.2, "", C["mem"], ec=C["star"], lw=2.4)
ax.text(6.1, 2.25, "★ Bounded motion-gated memory  ·  M slots, O(1)", ha="center",
        fontsize=10, fontweight="bold", color=C["star"])
ax.text(6.1, 1.75, "slots: (place_code $mem_p$, position $mem_\\mu$, importance)", ha="center", fontsize=8.5)
ax.text(6.1, 1.25, "content $=\\langle p,\\,mem_p\\rangle$     "
                   "gate $=-\\|\\mu-mem_\\mu\\|^2/\\tau^2$  ← motion-reachability", ha="center", fontsize=8.5)
ax.text(6.1, 0.78, "$w=\\mathrm{softmax}(\\,content + gate + imp\\,)$   →   $read_\\mu,\\ read_p$",
        ha="center", fontsize=8.5, fontweight="bold")

# ---- output
box(13.5, 5.3, 2.2, 1.7, "displacement\n$\\hat\\mu_t$ (2)\nfrom start", C["out"], 9.5)

# ---- arrows: forward path
arrow(2.6, 7.3, 3.2, 7.35)                                   # omni -> enc
arrow(5.8, 7.3, 6.6, 6.9)                                    # enc -> GRU   (p)
ax.text(6.15, 7.25, "$p$", fontsize=9, color=EC)
arrow(2.6, 2.85, 6.7, 5.3, rad=0.05, txt="motion", ty=0.25)  # motion -> GRU
arrow(9.0, 6.15, 9.7, 6.15)                                  # GRU -> mu update   (h)
ax.text(9.35, 6.35, "$h$", fontsize=9, color=EC)
arrow(12.8, 6.15, 13.5, 6.15)                                # mu update -> output

# ---- arrows: memory read/correct loop
arrow(4.6, 6.5, 4.6, 2.5, rad=0.0, color="#3d85c6", txt="$p$", tx=-0.25)        # p -> memory (content)
arrow(11.0, 5.3, 8.5, 2.5, rad=0.12, color="#3d85c6", txt="$\\mu$", tx=0.3)     # mu -> memory (gate)
arrow(9.2, 1.4, 11.0, 2.9, rad=-0.12, color="#cc4125",
      txt="$read_\\mu, read_p$", tx=1.7, ty=-0.1)                               # memory -> LC gate
arrow(11.25, 4.5, 11.25, 5.3, color="#cc4125")                                  # LC gate -> mu (corrected)
ax.text(11.55, 4.9, "corrected $\\mu$", fontsize=8, color="#cc4125", ha="left")

# ---- write-back (dashed)
arrow(12.6, 5.4, 9.2, 0.9, color="#999999", ls="--", rad=-0.25,
      txt="write $(p,\\mu)$  →  ring buffer (detach)", tx=-0.2, ty=0.35, tc="#999999")

# ---- legend of inductive biases -> diagnosis
ax.text(13.2, 3.7, "design = R2–R7 diagnosis", fontsize=9, fontweight="bold", color="#333")
for i, (t, col) in enumerate([
        ("rot-pool / MLP encoder  → R4 place code", C["enc"]),
        ("GRU integration  → R2 local state", C["gru"]),
        ("motion-gated read  → R6-C breakthrough", C["star"]),
        ("loop-closure gate  → soft SLAM closure", C["lc"]),
        ("M slots, O(1)  → R3 bounded memory", C["mem"])]):
    ax.add_patch(FancyBboxPatch((13.0, 3.25 - i * 0.42), 0.25, 0.25,
                 boxstyle="round,pad=0.02", fc=col, ec=EC, lw=1))
    ax.text(13.35, 3.37 - i * 0.42, t, fontsize=7.8, va="center")

plt.tight_layout()
plt.savefig("fig_mgm_arch.png", dpi=150, bbox_inches="tight")
print("saved fig_mgm_arch.png")
