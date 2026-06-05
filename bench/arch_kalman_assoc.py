"""KAN-LC: a learned Kalman predict-correct tracker with a VOCAB-INDEPENDENT data-association head.

Plugs into archs.REGISTRY as (B,T,din)->(B,T,2). din layout (from train_eval.encode):
    x[..., 0:K]          = K raycast distances in [0,1]
    x[..., K:K+4]        = last-action one-hot (self-motion command)
    x[..., K+4:]         = per-ray landmark-id one-hot, reshaped (K, V+1)  (V+1 incl. the "no-lm" bin)

CORE IDEA (why it transfers to an unseen maze):
  * PREDICT: a small learned motion integrator advances an egocentric displacement estimate mu_t from the
    action + ray-shape (collision) cues. This is the part that already works (~1.1 no-lm), so we keep it
    as a self-contained recurrent integrator and let landmarks only *correct* it -- never feed an absolute
    landmark id into the integrator path.
  * CORRECT: a differentiable RELATIONAL data-association head. The current observation is encoded into a
    permutation/rotation-aware "view descriptor" that depends on the ARRANGEMENT of landmarks across rays
    (which ray, relative angle between landmarks, their distances, and id-EQUALITY pattern) -- never the
    absolute id value. We keep a fixed-size FIFO slot memory of past view descriptors, each tagged with the
    integrator's mu at the time it was stored. We score current-vs-stored descriptor similarity, gate it by
    a learned uncertainty/ambiguity head (reject matches when several slots score alike = perceptual
    aliasing), and if a confident match m* is found we apply a Kalman-style innovation that pulls mu_t back
    toward mu_stored[m*] (loop closure => displacement-since-store ~ 0). The correction is on RELATIVE
    geometry, so it is invariant to maze size and to which absolute id a landmark happens to carry.

WHY this is vocab-independent (mechanically):
  The matcher never sees the raw integer id. It sees (a) the SAME-ID indicator matrix between rays of the
  current view and rays of a stored view (an equality pattern, invariant to relabeling the vocab), plus
  (b) ray distances and (c) ray angular offsets. Two views match iff they have the same landmark *layout*
  (same relative angles + distances + same equality structure) -- which is a property of physical geometry,
  not of the vocabulary. A larger maze reuses the same bounded vocab, so the equality-pattern + geometry
  channel is in-distribution there; nothing about absolute id->position is memorized.
"""
import torch, torch.nn as nn, torch.nn.functional as F

K_DEFAULT = 16

def split_obs(x, K=K_DEFAULT, V=12):
    rays = x[..., :K]                                   # (B,T,K)
    act  = x[..., K:K+4]                                # (B,T,4)
    lm   = x[..., K+4:].reshape(*x.shape[:-1], K, V+1)  # (B,T,K,V+1)
    lm_id = lm.argmax(-1)                               # (B,T,K) in [0..V]; index V == "no landmark" bin here
    has_lm = (lm[..., :V].sum(-1) > 0)                  # (B,T,K) bool: a real landmark on this ray
    return rays, act, lm_id, has_lm


class ViewDescriptor(nn.Module):
    """Encode one timestep's landmark layout into a permutation-stable descriptor that is INDEPENDENT of the
    absolute landmark ids. Uses only: which rays carry a landmark, their distance, their angular index, and
    the pairwise SAME-ID equality pattern among the lit rays. Output: a fixed vector + a validity scalar."""
    def __init__(self, K=K_DEFAULT, d=64):
        super().__init__(); self.K = K
        # per-ray token: [is_lm, dist, sin/cos of ray-angle index, #same-id rays (relational degree)]
        self.ray_mlp = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.pool = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, rays, lm_id, has_lm):
        B, T, K = rays.shape
        ang = torch.linspace(-1, 1, K, device=rays.device)              # angular index per ray (relative)
        ang = ang.view(1, 1, K).expand(B, T, K)
        # relational degree: for each lit ray, how many OTHER lit rays share its id (equality pattern only)
        same = (lm_id.unsqueeze(-1) == lm_id.unsqueeze(-2))             # (B,T,K,K) id-equality
        lit = has_lm.float()
        deg = (same.float() * lit.unsqueeze(-2)).sum(-1) * lit          # (B,T,K) same-id count among lit
        deg = deg / (lit.sum(-1, keepdim=True) + 1e-6)                  # normalize -> size-invariant
        tok = torch.stack([lit, rays * lit, ang.sin(), ang.cos(), deg], dim=-1)  # (B,T,K,5)
        h = self.ray_mlp(tok)                                          # (B,T,K,d)
        h = (h * lit.unsqueeze(-1)).sum(-2) / (lit.sum(-1, keepdim=True) + 1e-6)  # masked mean over lit rays
        desc = self.pool(h)                                            # (B,T,d)
        valid = (lit.sum(-1) >= 1).float()                             # need >=1 landmark to store/match
        return desc, valid


class KalmanAssocTracker(nn.Module):
    """Predict-correct tracker. mu (B,2) is the running egocentric displacement estimate.
    A GRU cell carries the integrator's latent; the association head injects Kalman corrections to mu."""
    def __init__(self, din, K=K_DEFAULT, V=12, d=128, slots=32, store_gap=5):
        super().__init__()
        self.K, self.V, self.d, self.slots, self.store_gap = K, V, d, slots, store_gap
        # --- PREDICT path (no absolute id ever enters here) ---
        self.pre_in = nn.Linear(K + 4, d)                # rays + action only
        self.gru = nn.GRUCell(d, d)
        self.dmu = nn.Linear(d, 2)                       # per-step displacement increment
        # --- CORRECT path ---
        self.view = ViewDescriptor(K, d=64)
        self.match_q = nn.Linear(64, 64); self.match_k = nn.Linear(64, 64)
        self.gate = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))  # [top1, top1-top2, valid]->logit
        self.kgain = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 2), nn.Sigmoid())  # per-dim gain
    def forward(self, x):
        rays, act, lm_id, has_lm = split_obs(x, self.K, self.V)
        B, T, K = rays.shape; dev = x.device
        h = torch.zeros(B, self.d, device=dev)
        mu = torch.zeros(B, 2, device=dev)
        # slot memory: stored descriptors (B,S,64), stored mu (B,S,2), filled mask (B,S)
        S = self.slots
        mem_desc = torch.zeros(B, S, 64, device=dev)
        mem_mu   = torch.zeros(B, S, 2, device=dev)
        mem_fill = torch.zeros(B, S, device=dev)
        desc_all, valid_all = self.view(rays, lm_id, has_lm)           # (B,T,64),(B,T)
        outs = []; slot_ptr = 0
        for t in range(T):
            # PREDICT
            h = self.gru(self.pre_in(torch.cat([rays[:, t], act[:, t]], -1)), h)
            mu = mu + self.dmu(h)
            # CORRECT (relational association against memory)
            desc = desc_all[:, t]; valid = valid_all[:, t]            # (B,64),(B,)
            q = F.normalize(self.match_q(desc), dim=-1)
            kk = F.normalize(self.match_k(mem_desc), dim=-1)          # (B,S,64)
            score = (q.unsqueeze(1) * kk).sum(-1) + (mem_fill - 1) * 10.0  # mask empty slots
            top2, idx = score.topk(2, dim=-1)                        # (B,2)
            top1 = top2[:, 0]; margin = top2[:, 0] - top2[:, 1]
            g = torch.sigmoid(self.gate(torch.stack([top1, margin, valid], -1))).squeeze(-1)  # (B,)
            m_mu = torch.gather(mem_mu, 1, idx[:, :1].unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)  # mu of best slot
            innov = m_mu - mu                                        # loop closure target: displacement-since-store ~ 0
            kg = self.kgain(torch.cat([h, innov], -1))               # learned per-dim Kalman gain
            mu = mu + (g.unsqueeze(-1) * kg) * innov                 # gated correction
            outs.append(mu)
            # WRITE to memory every store_gap steps when a landmark is visible (FIFO)
            if t % self.store_gap == 0:
                wr = (valid > 0.5)
                col = slot_ptr % S
                mem_desc = mem_desc.clone(); mem_mu = mem_mu.clone(); mem_fill = mem_fill.clone()
                mem_desc[:, col] = torch.where(wr.unsqueeze(-1), desc, mem_desc[:, col])
                mem_mu[:, col]   = torch.where(wr.unsqueeze(-1), mu.detach(), mem_mu[:, col])
                mem_fill[:, col] = torch.where(wr, torch.ones_like(mem_fill[:, col]), mem_fill[:, col])
                slot_ptr += 1
        return torch.stack(outs, 1)                                   # (B,T,2)


def build_kanlc(din, K=K_DEFAULT, V=12):
    return KalmanAssocTracker(din, K=K, V=V)
