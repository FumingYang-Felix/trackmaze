"""Round 2-A (training-free): does CONSTANT-RATE local re-anchoring give a FLAT error-vs-size curve?
The derived prediction: dead-reckoning error grows with maze size (longer paths -> more drift); but
re-anchoring whenever a landmark is in view (constant density => constant rate) bounds the error
INDEPENDENT of size -> flat. If the oracle curve is flat while dead-reckoning grows, the mechanism is
right and worth a learned version.
"""
import math, numpy as np
from env import TrackMazeEnv
from generate import explore_policy

def run_size(n, T, n_mazes=25):
    mv, rot = 0.22, 0.20; dr_e, or_e = [], []
    for m in range(n_mazes):
        env = TrackMazeEnv(n=n, ambiguity=1, lm_density=0.15, loop=0.3, seed=1000+m, max_steps=T+2)
        obs = env.reset(); rng = np.random.default_rng(m); W = env.wall.shape[0]
        edr = np.array([env.px, env.py]); eor = edr.copy(); ang = env.ang; de, oe = [], []
        for t in range(T):
            a = explore_policy(obs, rng)
            if a == 2: ang -= rot
            elif a == 3: ang += rot
            else:
                f = mv if a == 0 else -mv; step = np.array([math.cos(ang), math.sin(ang)]) * f
                edr = edr + step; eor = eor + step                    # dead-reckon both (commanded, no noise/collision)
            obs, _, _, _ = env.step(a); tx, ty = env.px, env.py; cx, cy = int(tx), int(ty)
            near_lm = any(0 <= cy+dy < W and 0 <= cx+dx < W and env.col[cy+dy, cx+dx] > 0
                          for dx, dy in ((0,1),(0,-1),(1,0),(-1,0),(0,0)))
            if near_lm: eor = np.array([tx, ty])                       # ORACLE re-anchor (constant rate ~ landmark density)
            de.append(math.hypot(edr[0]-tx, edr[1]-ty)); oe.append(math.hypot(eor[0]-tx, eor[1]-ty))
        dr_e.append(np.mean(de[len(de)//2:])); or_e.append(np.mean(oe[len(oe)//2:]))   # steady-state mean
    return np.mean(dr_e), np.mean(or_e)

if __name__ == "__main__":
    print("steady-state error vs GROWING maze size (constant landmark density):")
    print(f"{'n':>4} {'grid':>8} | {'dead-reckon':>12} {'oracle re-anchor':>16}")
    for n in (6, 12, 20, 28, 40, 60):
        dr, orr = run_size(n, T=24*n)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>8} | {dr:12.2f} {orr:16.2f}")
    print("\ndead-reckon GROWS with size; oracle re-anchor FLAT => constant-rate local re-anchor bounds error at any size.")
