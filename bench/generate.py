"""Generate trajectory datasets from TrackMazeEnv with a scripted explorer (corridor-following with
turns at junctions -> coverage + spatial loops, so re-anchoring is possible). Each episode exports the
egocentric obs stream + actions + full ground truth, for the localization probe and the metrics."""
import math
import numpy as np
from env import TrackMazeEnv

def explore_policy(obs, rng):
    rays = obs["rays"]; K = len(rays); ahead = rays[K//2]
    if ahead > 0.10 and rng.random() > 0.07:      # corridor: go forward (with small chance to turn)
        return 0
    left, right = rays[:K//2].mean(), rays[K//2:].mean()
    if abs(left-right) < 0.02: return 2 if rng.random() < 0.5 else 3
    return 2 if left > right else 3               # turn toward the more open side

def egocentric_disp(pos, ang0):
    """Rotate world displacement-from-start into the agent's own start-frame -> the localization target
    (what path-integration should recover). pos: (T,2), returns (T,2)."""
    d = pos - pos[0]; c, s = math.cos(-ang0), math.sin(-ang0)
    return np.stack([c*d[:,0] - s*d[:,1], s*d[:,0] + c*d[:,1]], axis=1).astype(np.float32)

def gen_dataset(n, n_eps, T, lm_density=0.15, ambiguity=0, seed=0, action_noise=0.03, rot_noise=0.09):
    rng = np.random.default_rng(seed)
    FEAT=[]; LM=[]; ACT=[]; POS=[]; DISP=[]; RA=[]; ANG0=[]; K=None; LBINS=None
    for e in range(n_eps):
        env = TrackMazeEnv(n=n, lm_density=lm_density, ambiguity=ambiguity, seed=seed*100000+e,
                           max_steps=T+2, action_noise=action_noise, rot_noise=rot_noise)
        obs = env.reset(); K=env.K; LBINS=env.LBINS
        f=[]; lm=[]; ac=[]; ra=[]; po=[np.array([env.px,env.py], dtype=np.float32)]   # po[0] = start pos
        ang0 = env.ang
        for t in range(T):
            a = explore_policy(obs, rng)
            f.append(np.concatenate([obs["rays"], obs["last_action"]]).astype(np.float32))  # egocentric obs + own command
            lm.append(obs["ray_lm"].astype(np.int32)); ac.append(a)
            obs, rew, done, gt = env.step(a)
            po.append(gt["pos"]); ra.append(gt["reanchor"])
        pos = np.stack(po[:T])
        FEAT.append(np.stack(f)); LM.append(np.stack(lm)); ACT.append(np.array(ac, dtype=np.int64))
        POS.append(pos); DISP.append(egocentric_disp(pos, ang0)); RA.append(np.array(ra[:T], dtype=bool)); ANG0.append(ang0)
    return dict(feat=np.stack(FEAT), ray_lm=np.stack(LM), action=np.stack(ACT),
                pos=np.stack(POS), disp=np.stack(DISP), reanchor=np.stack(RA),
                ang0=np.array(ANG0, dtype=np.float32), n=n, K=K, LBINS=LBINS)

if __name__ == "__main__":
    ds = gen_dataset(n=8, n_eps=4, T=120, seed=0)
    for k,v in ds.items():
        print(k, getattr(v,'shape',v))
