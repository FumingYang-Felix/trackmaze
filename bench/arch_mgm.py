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


class MGM(nn.Module):
    def __init__(self, h=128, d=64, M=64, tau=2.0):
        super().__init__()
        self.enc = RotInvEnc(d=d)
        self.gru = nn.GRUCell(d + 2, h)
        self.to_mu = nn.Linear(h, 2)                  # residual position head (refines integrated mu)
        self.gate = nn.Sequential(nn.Linear(h + d + 2, 32), nn.ReLU(), nn.Linear(32, 1))  # loop-closure confidence
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau)))))
        self.h, self.d, self.M = h, d, M

    def forward(self, geo, lm, motion):
        B, T, K = geo.shape
        dev = geo.device
        h = torch.zeros(B, self.h, device=dev)
        mu = torch.zeros(B, 2, device=dev)
        mem_p = torch.zeros(B, self.M, self.d, device=dev)
        mem_mu = torch.zeros(B, self.M, 2, device=dev)
        mem_mask = torch.zeros(B, self.M, device=dev)
        wptr = 0
        tau2 = (self.log_tau.exp() ** 2) + 1e-3
        outs = []
        for t in range(T):
            p = self.enc(geo[:, t], lm[:, t])                       # (B,d) heading-free place code
            h = self.gru(torch.cat([p, motion[:, t]], -1), h)
            mu = mu + motion[:, t] + self.to_mu(h)                  # integrate odometry + learned residual
            # motion-gated memory read
            content = (mem_p * p.unsqueeze(1)).sum(-1) / (self.d ** 0.5)        # (B,M)
            gatev = -((mem_mu - mu.unsqueeze(1)) ** 2).sum(-1) / tau2           # motion-reachability gate
            logits = content + gatev + (mem_mask - 1.0) * 1e4                   # mask empty slots
            w = torch.softmax(logits, 1)                                       # (B,M)
            read_mu = (w.unsqueeze(-1) * mem_mu).sum(1)                         # retrieved position
            read_p = (w.unsqueeze(-1) * mem_p).sum(1)                          # retrieved place code
            conf = torch.sigmoid(self.gate(torch.cat([h, read_p, read_mu - mu], -1)))  # (B,1)
            mu = mu + conf * (read_mu - mu)                                    # loop-closure correction
            outs.append(mu)
            # write to ring buffer (detach memory contents -> stable, like an episodic store)
            mem_p = mem_p.clone(); mem_mu = mem_mu.clone(); mem_mask = mem_mask.clone()
            mem_p[:, wptr] = p.detach(); mem_mu[:, wptr] = mu.detach(); mem_mask[:, wptr] = 1.0
            wptr = (wptr + 1) % self.M
        return torch.stack(outs, 1)                                            # (B,T,2)


class GRUBaseline(nn.Module):
    """Plain GRU on [geo,lm,motion] -> displacement (the phase-1 style baseline)."""
    def __init__(self, din=66, h=128):
        super().__init__(); self.gru = nn.GRU(din, h, batch_first=True); self.out = nn.Linear(h, 2)

    def forward(self, geo, lm, motion):
        x = torch.cat([geo, lm, motion], -1)
        y, _ = self.gru(x); return self.out(y)
