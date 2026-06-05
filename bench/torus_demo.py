"""Demo video: an agent path-integrates in a periodic arena (left) while its LEARNED LATENT traces the
TORUS (right). Uses the real trained torus model from toy_torus (latent verified b1=2 with ripser).
The latent's two phase coordinates (decoded from the model's readout) place it on a donut; when the agent
returns to a place, the latent returns to the same point on the torus = loop closure for free.

Saves torus_demo.gif (the video) + torus_demo_strip.png (3 stills).
"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from toy_torus import PIRNN, gen_walk, torus_emb, train

torch.manual_seed(0); np.random.seed(0)
print("training the torus model ...")
V, P = gen_walk(2000, 60, sig=0.12, seed=1)
m = PIRNN(h=128, periodic=True)
train(m, torch.tensor(V), torch.tensor(torus_emb(P)), torch.tensor(torus_emb(P[:, 0])), epochs=120)

print("rolling out one demo trajectory ...")
Vd, Pd = gen_walk(1, 150, sig=0.12, seed=42)
with torch.no_grad():
    pred, _ = m(torch.tensor(Vd), torch.tensor(torus_emb(Pd[:, 0])))
pred = pred[0].numpy()
thx = np.arctan2(pred[:, 1], pred[:, 0])              # latent phase coord 1 (around the big loop)
thy = np.arctan2(pred[:, 3], pred[:, 2])              # latent phase coord 2 (around the tube)
tx, ty = Pd[0, :, 0], Pd[0, :, 1]; T = len(thx)

R, r = 2.0, 0.7
def torus_xyz(u, v): return ((R + r*np.cos(v))*np.cos(u), (R + r*np.cos(v))*np.sin(u), r*np.sin(v))
uu, vv = np.meshgrid(np.linspace(0, 2*np.pi, 70), np.linspace(0, 2*np.pi, 28))
Xs, Ys, Zs = torus_xyz(uu, vv)

fig = plt.figure(figsize=(11, 5), facecolor="#0f1116")
ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2, projection="3d")

def draw(t):
    ax1.clear(); ax1.set_facecolor("#0a0b0f"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("agent walking the arena", color="#e6e9ef")
    ax1.plot(tx[:t+1], ty[:t+1], "-", color="#4c9be8", lw=1.2, alpha=0.5)
    ax1.plot(tx[t], ty[t], "o", color="#fff", ms=10)
    ax2.clear(); ax2.set_axis_off(); ax2.set_facecolor("#0f1116")
    ax2.set_title("its latent on the TORUS (b1=2)", color="#e6e9ef")
    ax2.plot_surface(Xs, Ys, Zs, color="#2a3140", alpha=0.18, linewidth=0, shade=True)
    lx, ly, lz = torus_xyz(thx[:t+1], thy[:t+1])
    ax2.plot(lx, ly, lz, "-", color="#e8954c", lw=1.6, alpha=0.7)
    px, py, pz = torus_xyz(thx[t], thy[t]); ax2.scatter([px], [py], [pz], color="#e8954c", s=70, depthshade=False)
    ax2.view_init(elev=32, azim=t*1.2); ax2.set_box_aspect((1, 1, 0.5))

print("rendering strip + gif ...")
fig2, axs = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0f1116",
                         subplot_kw=dict(projection="3d"))
for k, t in enumerate([T//4, T//2, T-1]):
    a = axs[k]; a.set_axis_off(); a.set_title(f"step {t}", color="#e6e9ef")
    a.plot_surface(Xs, Ys, Zs, color="#2a3140", alpha=0.18, linewidth=0)
    lx, ly, lz = torus_xyz(thx[:t+1], thy[:t+1]); a.plot(lx, ly, lz, "-", color="#e8954c", lw=1.6)
    px, py, pz = torus_xyz(thx[t], thy[t]); a.scatter([px], [py], [pz], color="#e8954c", s=70, depthshade=False)
    a.view_init(elev=32, azim=t*1.2); a.set_box_aspect((1, 1, 0.5))
fig2.suptitle("latent point tracing the torus as the agent walks", color="#e6e9ef")
fig2.tight_layout(); fig2.savefig("torus_demo_strip.png", dpi=110, facecolor="#0f1116")

anim = animation.FuncAnimation(fig, draw, frames=T, interval=70)
anim.save("torus_demo.gif", writer=animation.PillowWriter(fps=14))
print("saved torus_demo.gif + torus_demo_strip.png")
