"""RLC v2 — Relational Loop-Closure tracker with GT-BOUND alignment + calibrated confidence.

v1 lesson: leaving the model to discover *when* to trust a match (from position error alone) collapsed —
it learned to never correct, because a wrong match (perceptual aliasing) catastrophically corrupts the
persistent offset. v2 BINDS the two hard operations to ground truth during training (we have it):

  PREDICT (landmark-free):  2-layer GRU over [rays,last_action] -> integrated pose mu_prior (~the floor).
  PLACE CODE (relational):  contrastive view embedding v_t (same true cell -> similar; transfers to new mazes).
  CONFIDENCE (GT-bound):    a head reads the MATCH QUALITY (top cosine, top1-top2 gap, peakedness) and
                            outputs g in [0,1]; trained with BCE against the GT label "is the top-matched
                            slot really the same cell?" -> learns calibrated trust, suppresses aliasing.
  ALIGN (persistent):       on a trusted match, pull a running drift offset so the estimate snaps to the
                            retrieved earlier pose and CARRIES FORWARD (re-aligns the intrinsic map).

At inference there is no GT: the confidence head (now calibrated) decides when to align.
"""
import torch, torch.nn as nn, torch.nn.functional as F

K_DEFAULT, V_DEFAULT = 16, 12

def split_obs(x, K=K_DEFAULT, V=V_DEFAULT):
    rays = x[..., :K]; act = x[..., K:K+4]
    lm = x[..., K+4:].reshape(*x.shape[:-1], K, V+1)
    return rays, act, lm.argmax(-1)            # lm_id: 0="no landmark", 1..V = id+1

class RLCTracker(nn.Module):
    def __init__(self, din, K=K_DEFAULT, V=V_DEFAULT, d=128, store_gap=3, mask_recent=12, temp=0.07, tol=0.6):
        super().__init__()
        self.K, self.V, self.d, self.store_gap, self.mask_recent, self.temp, self.tol = K, V, d, store_gap, mask_recent, temp, tol
        self.gru = nn.GRU(K+4, d, num_layers=2, batch_first=True)              # PREDICT (landmark-free)
        self.pose = nn.Linear(d, 2)
        self.lm_emb = nn.Embedding(V+1, 16)
        self.view = nn.Sequential(nn.Linear(K + K*16, d), nn.GELU(), nn.Linear(d, d))   # relational place code
        self.conf = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))         # confidence from match quality
        self.use_memory = True
    def encode_view(self, rays, lm_id):
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
        offset = torch.zeros(B, 2, device=dev)
        g_list, lbl_list = [], []
        ar = torch.arange(B, device=dev)
        for t in range(T):
            q = v[:, t]
            while n_readable < len(times) and times[n_readable] <= t - self.mask_recent:
                n_readable += 1
            if n_readable > 0:
                Kmem = torch.stack(keys[:n_readable], 1)                    # (B,S,d) detached, unit-norm
                Vmem = torch.stack(vals[:n_readable], 1)                    # (B,S,2) detached
                cos = torch.bmm(Kmem, q.unsqueeze(-1)).squeeze(-1)          # (B,S) cosine (q unit-norm)
                S = cos.shape[1]
                attn = torch.softmax(cos / self.temp, dim=1)
                j = attn.argmax(1)                                         # top slot
                top = torch.topk(cos, min(2, S), dim=1).values
                c1 = top[:, 0]; c2 = top[:, 1] if S >= 2 else top[:, 0]
                feats = torch.stack([c1, c1 - c2, attn.max(1).values], -1) # match-quality features
                g = torch.sigmoid(self.conf(feats)).squeeze(-1)           # calibrated trust in [0,1]
                retrieved = Vmem[ar, j]                                    # the trusted earlier estimate
                offset = (1 - g).unsqueeze(-1) * offset + g.unsqueeze(-1) * (retrieved - mu_prior[:, t])
                if gt_disp is not None:                                    # bind trust to: will this correction HELP?
                    label = (torch.norm(retrieved - gt_disp[:, t], dim=-1) < self.tol).float()  # retrieved ~ true pos?
                    g_list.append(g); lbl_list.append(label)
            mu_t = mu_prior[:, t] + offset
            mu_out.append(mu_t)
            if t % self.store_gap == 0:
                keys.append(v[:, t].detach()); vals.append(mu_t.detach()); times.append(t)
        mu = torch.stack(mu_out, 1)
        if return_aux:
            g_all = torch.stack(g_list) if g_list else None
            lbl_all = torch.stack(lbl_list) if lbl_list else None
            return mu, v, mu_prior, g_all, lbl_all
        return mu

def build_rlc(din): return RLCTracker(din)
