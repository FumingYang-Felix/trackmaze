"""Round 8: the LEARNED architecture, designed from the R2-R7 diagnosis. A new RNN-type arch + training
trick for OOD-size state tracking. Inductive biases (each = a diagnosed size-invariant ingredient):

  1. ROTATION-INVARIANT obs encoder (R4): circular-conv over the 32-ray omni view, pooled over rotations ->
     a heading-free place code that transfers across size and viewpoint.
  2. RECURRENT LOCAL INTEGRATION (R2): a GRU integrates the place code + commanded motion -> size-invariant
     local state and a running position estimate mu (dead-reckoning, drifts).
  3. MOTION-GATED MEMORY READ (R6-C, the breakthrough, made soft+differentiable): a bounded memory of past
     (place_code, position) pairs. The read attends by content similarity GATED by motion-reachability
     exp(-||mu_t - mu_i||^2 / tau^2) -- so it can only re-anchor to places its motion estimate says are
     near, NOT to far look-alikes (which is exactly what made global appearance matching collapse). A learned
     gate applies the retrieved position as a loop-closure correction to mu.
  4. BOUNDED memory (R3): a fixed ring buffer of M slots -> O(1), size-independent.

Predicts egocentric displacement-from-start. The thesis: trained on small mazes it tracks large ones far
better than a plain GRU/Transformer (which memorize a landmark vocabulary that doesn't transfer -- phase-1).
"""
import torch, torch.nn as nn, torch.nn.functional as F


class RotInvEnc(nn.Module):
    def __init__(self, K=32, c=32, d=64):
        super().__init__(); self.K = K
        self.c1 = nn.Conv1d(2, c, 5, padding=2, padding_mode="circular")
        self.c2 = nn.Conv1d(c, c, 5, padding=2, padding_mode="circular")
        self.head = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, geo, lm):
        z = torch.stack([geo, lm], 1)                 # (B,2,K)
        z = F.relu(self.c1(z)); z = F.relu(self.c2(z))
        z = z.max(dim=2).values                       # rotation pooling -> heading-free
        return self.head(z)                           # (B,d)


class PlainEnc(nn.Module):
    """Non-rotation-invariant MLP encoder (ablation): flattens the 64-d view -> place code (can memorize
    heading/landmark configs -> the phase-1 trap)."""
    def __init__(self, din=64, d=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, geo, lm):
        return self.net(torch.cat([geo, lm], -1))


class MGM(nn.Module):
    def __init__(self, h=128, d=64, M=64, tau=2.0, use_rot=True, use_mem=True, use_grid=False,
                 use_kv=False, n_corr=1, dk=48, periods=(4.0, 8.0, 16.0, 32.0)):
        super().__init__()
        self.use_mem = use_mem; self.use_grid = use_grid; self.use_kv = use_kv; self.n_corr = n_corr
        self.register_buffer("periods", torch.tensor(periods).float())   # idea-B: multi-scale periodic gate
        self.enc = RotInvEnc(d=d) if use_rot else PlainEnc(d=d)
        self.gru = nn.GRUCell(d + 2, h)
        self.to_mu = nn.Linear(h, 2)                  # residual position head (refines integrated mu)
        self.gate = nn.Sequential(nn.Linear(h + d + 2, 32), nn.ReLU(), nn.Linear(32, 1))  # loop-closure confidence
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau)))))
        self.h, self.d, self.M, self.dk = h, d, M, dk
        if use_kv:
            self.key = nn.Linear(d, dk); self.qry = nn.Linear(d, dk)        # learned key/query projections
            self.wimp = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 1))  # learned write importance (R3)

    def init_state(self, B, dev):
        return dict(h=torch.zeros(B, self.h, device=dev), mu=torch.zeros(B, 2, device=dev),
                    mem_p=torch.zeros(B, self.M, self.d, device=dev),
                    mem_k=torch.zeros(B, self.M, self.dk, device=dev) if self.use_kv else None,
                    mem_mu=torch.zeros(B, self.M, 2, device=dev),
                    mem_occ=torch.zeros(B, self.M, device=dev), mem_imp=torch.zeros(B, self.M, device=dev), wptr=0)

    def step(self, geo_t, lm_t, motion_t, s):
        """One online timestep. Mutates state dict s; returns mu (B,2)."""
        tau2 = (self.log_tau.exp() ** 2) + 1e-3
        p = self.enc(geo_t, lm_t)
        s["h"] = self.gru(torch.cat([p, motion_t], -1), s["h"])
        mu = s["mu"] + motion_t + self.to_mu(s["h"])
        if self.use_mem:
            if self.use_kv:
                content = (s["mem_k"] * self.qry(p).unsqueeze(1)).sum(-1) / (self.dk ** 0.5)
            else:
                content = (s["mem_p"] * p.unsqueeze(1)).sum(-1) / (self.d ** 0.5)
            for _ in range(self.n_corr):
                if self.use_grid:
                    diff = s["mem_mu"] - mu.unsqueeze(1)
                    ph = 2 * 3.141592653589793 * diff.unsqueeze(-1) / self.periods.view(1, 1, 1, -1)
                    gatev = torch.cos(ph).mean((-1, -2)) * 4.0
                else:
                    gatev = -((s["mem_mu"] - mu.unsqueeze(1)) ** 2).sum(-1) / tau2
                w = torch.softmax(content + gatev + s["mem_imp"] + (s["mem_occ"] - 1.0) * 1e4, 1)
                read_mu = (w.unsqueeze(-1) * s["mem_mu"]).sum(1)
                read_p = (w.unsqueeze(-1) * s["mem_p"]).sum(1)
                conf = torch.sigmoid(self.gate(torch.cat([s["h"], read_p, read_mu - mu], -1)))
                mu = mu + conf * (read_mu - mu)
            wp = s["wptr"]
            # clone before writing: the read above saves these buffers for backward; in-place would corrupt it
            s["mem_p"] = s["mem_p"].clone(); s["mem_mu"] = s["mem_mu"].clone()
            s["mem_occ"] = s["mem_occ"].clone(); s["mem_imp"] = s["mem_imp"].clone()
            s["mem_p"][:, wp] = p.detach(); s["mem_mu"][:, wp] = mu.detach(); s["mem_occ"][:, wp] = 1.0
            if self.use_kv:
                s["mem_k"] = s["mem_k"].clone(); s["mem_k"][:, wp] = self.key(p).detach()
                s["mem_imp"][:, wp] = torch.log(torch.sigmoid(self.wimp(s["h"])).squeeze(-1) + 1e-4).detach()
            s["wptr"] = (wp + 1) % self.M
        s["mu"] = mu
        return mu

    def forward(self, geo, lm, motion):
        B, T, K = geo.shape
        s = self.init_state(B, geo.device); outs = []
        for t in range(T):
            outs.append(self.step(geo[:, t], lm[:, t], motion[:, t], s))
        return torch.stack(outs, 1)                                            # (B,T,2)


class GRUBaseline(nn.Module):
    """Plain GRU on [geo,lm,motion] -> displacement (the phase-1 style baseline)."""
    def __init__(self, din=66, h=128):
        super().__init__(); self.gru = nn.GRU(din, h, batch_first=True); self.out = nn.Linear(h, 2)

    def forward(self, geo, lm, motion):
        x = torch.cat([geo, lm, motion], -1)
        y, _ = self.gru(x); return self.out(y)


class LSTMBaseline(nn.Module):
    def __init__(self, din=66, h=128):
        super().__init__(); self.lstm = nn.LSTM(din, h, batch_first=True); self.out = nn.Linear(h, 2)

    def forward(self, geo, lm, motion):
        y, _ = self.lstm(torch.cat([geo, lm, motion], -1)); return self.out(y)


class TransformerBaseline(nn.Module):
    """Causal Transformer on [geo,lm,motion] -> displacement (phase-1 baseline; expected to memorize)."""
    def __init__(self, din=66, h=128, layers=4, heads=4, maxT=2200):
        super().__init__()
        self.inp = nn.Linear(din, h)
        self.pos = nn.Parameter(torch.zeros(1, maxT, h))
        enc = nn.TransformerEncoderLayer(h, heads, h * 2, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers); self.out = nn.Linear(h, 2)

    def forward(self, geo, lm, motion):
        x = torch.cat([geo, lm, motion], -1); B, T, _ = x.shape
        x = self.inp(x) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        return self.out(self.tr(x, mask=mask))


def build(arch):
    if arch == "gru": return GRUBaseline()
    if arch == "lstm": return LSTMBaseline()
    if arch == "transformer": return TransformerBaseline()
    if arch == "mgm": return MGM(use_rot=True, use_mem=True)
    if arch == "mgm_norot": return MGM(use_rot=False, use_mem=True)     # ablate rotation-invariance (Round 8 best)
    if arch == "mgm_nomem": return MGM(use_rot=True, use_mem=False)     # ablate motion-gated memory
    if arch == "mgm_grid": return MGM(use_rot=False, use_mem=True, use_grid=True)   # idea-B periodic gate
    if arch == "mgm_grid_M128": return MGM(use_rot=False, use_mem=True, use_grid=True, M=128)
    # Round 10 (D): strengthen -- learned key/value memory + learned write importance + iterative correction
    if arch == "mgm_v2": return MGM(use_rot=False, use_mem=True, use_kv=True, n_corr=2, h=192, M=96)
    if arch == "mgm_v2_big": return MGM(use_rot=False, use_mem=True, use_kv=True, n_corr=3, h=256, M=128)
    raise ValueError(arch)
