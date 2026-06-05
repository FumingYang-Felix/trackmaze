"""RLC: Relational Loop-Closure tracker — the design from the human-vs-model diagnosis.

The phase-1 failure: feeding landmark ids into the tracker let it MEMORIZE config->position (a
non-transferable shortcut) which corrupted OOD integration. Fix = separate the two pathways and make the
correction RELATIONAL + RETRIEVAL-based, like episodic loop-closure rather than stimulus->response:

  PREDICT (landmark-FREE):   a GRU over [rays, last_action] integrates an egocentric pose estimate mu_prior.
                             Landmarks NEVER touch this stream, so they can't corrupt integration (~1.1 floor).
  VIEW CODE (relational):    a place embedding v_t from rays + landmark embeddings, trained by a CONTRASTIVE
                             loss to be SIMILAR iff the same true cell (id used relationally, not memorized).
  EPISODIC MEMORY + CORRECT: write (v_t, mu_t) into a slot store; query the current v_t against PAST slots
                             (older than mask_recent steps); if a confident match is found (a revisit /
                             loop closure), pull mu toward the retrieved earlier estimate -> drift resets to
                             its (smaller) value at the earlier visit. A gate suppresses correction when no
                             match (novel place / perceptual aliasing) -> falls back to mu_prior.

Worst case (gate off) => mu_prior, i.e. the ~1.1 integrator floor, already beating every with-landmark
baseline (1.6-2.9). If the contrastive matching transfers, error moves toward the oracle (~0.4).
"""
import torch, torch.nn as nn, torch.nn.functional as F

K_DEFAULT, V_DEFAULT = 16, 12

def split_obs(x, K=K_DEFAULT, V=V_DEFAULT):
    rays = x[..., :K]; act = x[..., K:K+4]
    lm = x[..., K+4:].reshape(*x.shape[:-1], K, V+1)
    lm_id = lm.argmax(-1)                      # 0 = "no landmark" bin, 1..V = landmark id+1
    return rays, act, lm_id

class RLCTracker(nn.Module):
    def __init__(self, din, K=K_DEFAULT, V=V_DEFAULT, d=128, store_gap=3, mask_recent=12, temp=0.07):
        super().__init__()
        self.K, self.V, self.d, self.store_gap, self.mask_recent, self.temp = K, V, d, store_gap, mask_recent, temp
        self.gru = nn.GRU(K+4, d, num_layers=2, batch_first=True)   # PREDICT: landmark-free integrator
        self.pose = nn.Linear(d, 2)
        self.lm_emb = nn.Embedding(V+1, 16)                  # shared id embedding (contrastive forces relational use)
        self.view = nn.Sequential(nn.Linear(K + K*16, d), nn.GELU(), nn.Linear(d, d))   # relational place code
        self.gate = nn.Linear(d, 1); self.use_memory = True
        self.gate_mode = "learned"; self.gate_thresh = 0.5   # eval can override: "match" | "hard"
        self.diag = False; self.diag_stats = {}
    def encode_view(self, rays, lm_id):
        e = self.lm_emb(lm_id).flatten(-2)                   # (B,T,K*16)
        return F.normalize(self.view(torch.cat([rays, e], -1)), dim=-1)   # unit-norm place embedding
    def forward(self, x, return_aux=False):
        rays, act, lm_id = split_obs(x, self.K, self.V)
        h,_ = self.gru(torch.cat([rays, act], -1)); mu_prior = self.pose(h)   # (B,T,2)
        v = self.encode_view(rays, lm_id)                                     # (B,T,d)
        if not self.use_memory:
            return (mu_prior, v, mu_prior) if return_aux else mu_prior
        B, T, _ = mu_prior.shape; dev = x.device
        mu_out = []
        keys, vals, times = [], [], []                   # detached episodic store (no in-place graph ops)
        n_readable = 0                                    # writes older than mask_recent are readable
        offset = torch.zeros(B, 2, device=dev)            # PERSISTENT running drift correction (carries forward)
        dmatch, dconf, dcorr = [], [], []                 # diagnostics
        for t in range(T):
            q = v[:, t]                                   # (B,d)
            while n_readable < len(times) and times[n_readable] <= t - self.mask_recent:
                n_readable += 1
            if n_readable > 0:
                Kmem = torch.stack(keys[:n_readable], 1)                       # (B,S,d) detached
                Vmem = torch.stack(vals[:n_readable], 1)                       # (B,S,2) detached
                sim = torch.bmm(Kmem, q.unsqueeze(-1)).squeeze(-1) / self.temp # (B,S)
                attn = torch.softmax(sim, dim=1)
                retrieved = torch.bmm(attn.unsqueeze(1), Vmem).squeeze(1)      # (B,2) matched earlier estimate
                match = attn.max(1).values                                     # match strength (softmax peak)
                if self.gate_mode == "match":  g = match                       # eval: trust the match directly
                elif self.gate_mode == "hard": g = (match > self.gate_thresh).float()
                else: g = torch.sigmoid(self.gate(q)).squeeze(-1) * match       # learned (train)
                # update the persistent offset so the corrected estimate -> retrieved, and CARRIES FORWARD
                target_off = retrieved - mu_prior[:, t]
                offset = (1 - g).unsqueeze(-1) * offset + g.unsqueeze(-1) * target_off
                if self.diag:
                    dmatch.append(match.mean().item()); dconf.append((match > 0.5).float().mean().item())
                    dcorr.append((g.unsqueeze(-1) * (target_off - offset)).norm(dim=-1).mean().item())
            mu_t = mu_prior[:, t] + offset                # estimate = integrator + running correction
            mu_out.append(mu_t)
            if t % self.store_gap == 0:                   # write the CORRECTED estimate (detached buffer)
                keys.append(v[:, t].detach()); vals.append(mu_t.detach()); times.append(t)
        mu = torch.stack(mu_out, 1)
        if self.diag and dmatch:
            self.diag_stats = {"avg_match": sum(dmatch)/len(dmatch), "frac_match>0.5": sum(dconf)/len(dconf),
                               "avg_correction_mag": sum(dcorr)/len(dcorr)}
        return (mu, v, mu_prior) if return_aux else mu

def build_rlc(din): return RLCTracker(din)
