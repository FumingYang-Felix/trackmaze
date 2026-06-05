"""TrackMaze env — the Python port of the demo geometry, as a trainable benchmark.

Egocentric, partial-observation 2.5D maze. The agent gets ONLY: K raycast distances + the landmark
id each ray hits + its last action (self-motion) + its heading. It does NOT observe its position.
Ground truth (position, heading, map, landmark ids, re-anchor events) is exported separately, for
supervision and for the drift/correction/generalization metrics.

Controllable difficulty: size n (the OOD axis) · landmark density · ambiguity (landmark-id vocabulary
size: 0 -> every landmark unique; higher -> aliasing). Drift source: the agent integrates its commanded
actions, but collisions (and optional action noise) make commanded != true motion -> open-loop drift,
correctable by recognizing landmarks (loop closure).
"""
import math
from collections import Counter
import numpy as np

DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

def gen_maze(n, lm_density=0.15, ambiguity=0, rng=None):
    """Randomized-DFS maze on an n x n cell grid -> (2n+1) wall grid + landmark-id grid.
    Returns wall (1=wall,0=open), col (0=plain wall / non-wall; >0 = landmark id+1), LBINS (vocab)."""
    rng = rng if rng is not None else np.random.default_rng()
    W = 2*n+1
    wall = np.ones((W,W), dtype=np.int8); col = np.zeros((W,W), dtype=np.int32)
    vis = np.zeros((n,n), dtype=bool); st=[(0,0)]; vis[0,0]=True; wall[1,1]=0
    while st:
        cx,cy = st[-1]; moved=False
        for oi in rng.permutation(4):
            dx,dy = DIRS[oi]; nx,ny = cx+dx, cy+dy
            if 0<=nx<n and 0<=ny<n and not vis[ny,nx]:
                wall[2*cy+1+dy, 2*cx+1+dx]=0; wall[2*ny+1, 2*nx+1]=0
                vis[ny,nx]=True; st.append((nx,ny)); moved=True; break
        if not moved: st.pop()
    ys,xs = np.where(wall==1); pick = rng.random(len(ys)) < lm_density
    lm = list(zip(ys[pick].tolist(), xs[pick].tolist())); rng.shuffle(lm)
    LBINS = max(1,len(lm)) if ambiguity==0 else max(1, min(len(lm), [0,12,5,2][ambiguity]))
    for i,(y,x) in enumerate(lm): col[y,x] = 1 + (i % LBINS)
    return wall, col, LBINS

def raycast(wall, col, px, py, ang, K=16, fov=math.pi/2.6, maxd=24.0):
    """Cast K rays in the FOV. Returns dist (K, normalized to [0,1]) and lmid (K, landmark id hit, -1=plain)."""
    dx,dy = math.cos(ang), math.sin(ang); pX,pY = -dy*math.tan(fov/2), dx*math.tan(fov/2)
    Wg = wall.shape[0]; dist=np.ones(K,dtype=np.float32); lmid=np.full(K,-1,dtype=np.int32)
    for i in range(K):
        cam = 2*(i+0.5)/K - 1; rdx=dx+pX*cam; rdy=dy+pY*cam
        mX=int(px); mY=int(py)
        ddx = abs(1/rdx) if rdx!=0 else 1e9; ddy = abs(1/rdy) if rdy!=0 else 1e9
        if rdx<0: sx=-1; sdx=(px-mX)*ddx
        else:     sx=1;  sdx=(mX+1-px)*ddx
        if rdy<0: sy=-1; sdy=(py-mY)*ddy
        else:     sy=1;  sdy=(mY+1-py)*ddy
        side=0; hit=False
        for _ in range(256):
            if sdx<sdy: sdx+=ddx; mX+=sx; side=0
            else:       sdy+=ddy; mY+=sy; side=1
            if not (0<=mY<Wg and 0<=mX<Wg): break
            if wall[mY,mX] > 0: hit=True; break
        if hit:
            pd = (sdx-ddx) if side==0 else (sdy-ddy)
            dist[i] = min(pd, maxd)/maxd; lmid[i] = col[mY,mX]-1   # -1 if plain wall
    return dist, lmid

class TrackMazeEnv:
    """Egocentric maze env. obs = {rays(K), ray_lm(K), last_action(4), heading(2)}; gt exported per step."""
    N_ACT = 4   # 0 forward, 1 back, 2 turn-left, 3 turn-right
    def __init__(self, n=8, lm_density=0.15, ambiguity=0, seed=0, K=16, max_steps=300,
                 action_noise=0.03, rot_noise=0.09, reanchor_gap=15):
        # action_noise: stdev of translation odometry noise (cells). rot_noise: stdev of PERSISTENT
        # per-step heading drift (rad) -> the latent that compounds. Neither is observable by the agent;
        # this is the drift it must correct by recognizing landmarks. Set both 0 for a noiseless oracle.
        # reanchor_gap: a step is a re-anchor OPPORTUNITY when the agent returns (with a landmark in view)
        # to a cell first visited >= this many steps ago (loop closure) -- vocabulary-independent.
        self.n,self.K,self.max_steps,self.reanchor_gap = n,K,max_steps,reanchor_gap
        self.action_noise, self.rot_noise = action_noise, rot_noise
        self.rng = np.random.default_rng(seed)
        self.wall, self.col, self.LBINS = gen_maze(n, lm_density, ambiguity, self.rng)
        ids = (self.col[self.col>0]-1).tolist(); cnt = Counter(ids)
        self.unique_ids = set(k for k,v in cnt.items() if v==1)   # landmarks that pin location (informative re-anchor)
        self.reset()
    def reset(self):
        self.px,self.py = 1.5,1.5; self.ang = float(self.rng.random()*2*math.pi); self.t=0
        self.visited=set(); self.cell_first={(1,1):0}   # cell -> first-visit step (for loop-closure re-anchor)
        ys,xs = np.where(self.wall==0); d=(xs-1.5)**2+(ys-1.5)**2; far = np.argsort(-d)[:6]; pk = far[self.rng.integers(len(far))]
        self.gx,self.gy = float(xs[pk]+0.5), float(ys[pk]+0.5)
        return self._obs(0)
    def _obs(self, last_action):
        # The agent observes ONLY: rays, the landmark id each ray hits, and its own last command.
        # No true position/heading (those would leak the state it must track).
        dist,lmid = raycast(self.wall, self.col, self.px, self.py, self.ang, self.K)
        a = np.zeros(self.N_ACT, dtype=np.float32); a[last_action]=1.0
        return {"rays":dist, "ray_lm":lmid, "last_action":a}
    def step(self, action):
        mv,rot = 0.22,0.20
        if action==2:   self.ang -= rot
        elif action==3: self.ang += rot
        else:
            f = mv if action==0 else -mv
            if self.action_noise: f += float(self.rng.normal(0,self.action_noise))
            nx = self.px+math.cos(self.ang)*f; ny = self.py+math.sin(self.ang)*f
            if self.wall[int(self.py),int(nx)]==0: self.px=nx      # collision -> no move (but agent commanded it -> drift)
            if self.wall[int(ny),int(self.px)]==0: self.py=ny
        if self.rot_noise: self.ang += float(self.rng.normal(0,self.rot_noise))  # persistent heading drift (the compounding latent)
        self.t += 1; cell = (int(self.px),int(self.py)); self.visited.add(cell)
        obs = self._obs(action)
        # re-anchor opportunity: a landmark is in view AND we're back in a cell first seen >= reanchor_gap ago
        sees_lm = bool((obs["ray_lm"] >= 0).any())
        loop_closure = cell in self.cell_first and (self.t - self.cell_first[cell]) >= self.reanchor_gap
        if cell not in self.cell_first: self.cell_first[cell] = self.t
        reanchor = sees_lm and loop_closure
        reached = math.hypot(self.gx-self.px, self.gy-self.py) < 0.5
        gt = {"pos":np.array([self.px,self.py],dtype=np.float32), "reanchor":bool(reanchor),
              "goal":np.array([self.gx,self.gy],dtype=np.float32),
              "heading":np.array([math.cos(self.ang),math.sin(self.ang)],dtype=np.float32)}
        return obs, (1.0 if reached else 0.0), bool(reached or self.t>=self.max_steps), gt
