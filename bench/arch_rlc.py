"""RLC v3 — configurable loop-closure correction back-end (sweepable).

v1/v2 isolated two residual problems: (a) the confidence head can't tell a HELPFUL correction from a
mere look-alike from cosine stats alone; (b) snapping to a single stored estimate transfers that
estimate's own drift, and one wrong match catastrophically corrupts a persistent offset. v3 makes the
correction back-end configurable so we can sweep the fixes:

  retr   : which stored pose to trust   -> 'best' (top cosine) | 'oldest' (least-drifted of top-k) | 'topk' (mean of top-k)
  update : how the running offset moves  -> 'snap' ((1-g)o+g*delta) | 'ema' (o += gain*g*(delta-o), outlier-robust)
  feats  : confidence input              -> 'basic' (cosine/gap/peak) | 'rich' (+ match AGE + top-k pose CONSISTENCY)
  gain   : global correction strength

Confidence is GT-bound at train time (label = "is the retrieved pose actually near the true position?").
Predict (landmark-free 2-layer GRU) and the relational contrastive place code are unchanged.
"""
import torch, torch.nn as nn, torch.nn.functional as F

K_DEFAULT, V_DEFAULT = 16, 12

def split_obs(x, K=K_DEFAULT, V=V_DEFAULT):
    rays = x[..., :K]; act = x[..., K:K+4]
    lm = x[..., K+4:].reshape(*x.shape[:-1], K, V+1)
    return rays, act, lm.argmax(-1)

class RLCTracker(nn.Module):
    def __init__(self, din, K=K_DEFAULT, V=V_DEFAULT, d=128, store_gap=3, mask_recent=12, temp=0.07, tol=0.6,
                 retr="best", update="snap", feats="basic", gain=1.0):
        super().__init__()
        self.K, self.V, self.d, self.store_gap, self.mask_recent, self.temp, self.tol = K, V, d, store_gap, mask_recent, temp, tol
        self.retr, self.update, self.feats, self.gain = retr, update, feats, gain
        self.gru = nn.GRU(K+4, d, num_layers=2, batch_first=True)
        self.pose = nn.Linear(d, 2)
        self.lm_emb = nn.Embedding(V+1, 16)
        self.view = nn.Sequential(nn.Linear(K + K*16, d), nn.GELU(), nn.Linear(d, d))     # absolute-id place code
        self.view_rel = nn.Sequential(nn.Linear(3*K, d), nn.GELU(), nn.Linear(d, d))      # permutation-INVARIANT place code
        self.conf = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))   # 5 features (basic zero-pads last 2)
        self.use_memory = True; self.relational = False
    def encode_view(self, rays, lm_id):
        if self.relational:
            has = (lm_id >= 0).float()                                            # (B,T,K) ray sees a landmark
            same = (((lm_id.unsqueeze(-1) == lm_id.unsqueeze(-2)) &
                     (lm_id.unsqueeze(-1) >= 0)).float().sum(-1) - has) / self.K  # #other rays with SAME id (id-invariant)
            return F.normalize(self.view_rel(torch.cat([rays, has, same], -1)), dim=-1)
        e = self.lm_emb(lm_id).flatten(-2)
        return F.normalize(self.view(torch.cat([rays, e], -1)), dim=-1)
    def forward(self, x, gt_disp=None, return_aux=False):
        rays, act, lm_id = split_obs(x, self.K, self.V)
        h, _ = self.gru(torch.cat([rays, act], -1)); mu_prior = self.pose(h)
        v = self.encode_view(rays, lm_id)
        B, T, _ = mu_prior.shape; dev = x.device
        if not self.use_memory:
            return (mu_prior, v, mu_prior, None, None) if return_aux else mu_prior
        mu_out, keys, vals, times, n_readable = [], [], [], [], 0
        offset = torch.zeros(B, 2, device=dev); ar = torch.arange(B, device=dev)
        g_list, lbl_list = [], []
        for t in range(T):
            q = v[:, t]
            while n_readable < len(times) and times[n_readable] <= t - self.mask_recent:
                n_readable += 1
            if n_readable > 0:
                Kmem = torch.stack(keys[:n_readable], 1)                  # (B,S,d)
                Vmem = torch.stack(vals[:n_readable], 1)                  # (B,S,2)
                tw = torch.tensor(times[:n_readable], device=dev, dtype=torch.float32)   # (S,) write times
                cos = torch.bmm(Kmem, q.unsqueeze(-1)).squeeze(-1)        # (B,S)
                S = cos.shape[1]; k = min(3, S)
                topv, topi = torch.topk(cos, k, dim=1)                    # (B,k)
                Vtop = torch.gather(Vmem, 1, topi.unsqueeze(-1).expand(-1, -1, 2))       # (B,k,2)
                ages = (t - tw[topi]) / max(T, 1)                         # (B,k) normalized recency
                attn = torch.softmax(cos / self.temp, dim=1)
                if   self.retr == "oldest": retrieved = Vtop[ar, ages.argmax(1)]         # least-drifted of the good matches
                elif self.retr == "topk":   retrieved = Vtop.mean(1)                     # consensus of top-k
                else:                        retrieved = Vtop[:, 0]                       # best cosine
                spread = Vtop.std(1).mean(-1) if k >= 2 else torch.zeros(B, device=dev)  # top-k pose agreement
                gap = (topv[:, 0] - topv[:, 1]) if k >= 2 else torch.zeros(B, device=dev)
                f = torch.stack([topv[:, 0], gap, attn.max(1).values,
                                 (-spread if self.feats == "rich" else torch.zeros(B, device=dev)),
                                 (ages[:, 0] if self.feats == "rich" else torch.zeros(B, device=dev))], -1)
                g = torch.sigmoid(self.conf(f)).squeeze(-1) * self.gain
                delta = retrieved - mu_prior[:, t]
                if self.update == "ema": offset = offset + 0.3 * g.unsqueeze(-1) * (delta - offset)
                else:                    offset = (1 - g).unsqueeze(-1) * offset + g.unsqueeze(-1) * delta
                if gt_disp is not None:
                    label = (torch.norm(retrieved - gt_disp[:, t], dim=-1) < self.tol).float()
                    g_list.append(g.clamp(1e-6, 1 - 1e-6)); lbl_list.append(label)
            mu_t = mu_prior[:, t] + offset
            mu_out.append(mu_t)
            if t % self.store_gap == 0:
                keys.append(v[:, t].detach()); vals.append(mu_t.detach()); times.append(t)
        mu = torch.stack(mu_out, 1)
        if return_aux:
            return mu, v, mu_prior, (torch.stack(g_list) if g_list else None), (torch.stack(lbl_list) if lbl_list else None)
        return mu

def build_rlc(din): return RLCTracker(din)
