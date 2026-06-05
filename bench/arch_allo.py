"""AlloTracker (Stage 1): integrator + allo-frame place matcher + EMA phase-lock correction.

  integrator : 2-layer GRU over [canon_geo(32), action(4)] -> egocentric displacement mu_prior (the floor).
  place code : MLP over the ALLO-canonical view [canon_geo(32), canon_lm(32)] -> unit-norm v_t.
               This is the validated heading-invariant descriptor (same-cell vs diff-cell gap ~0.8).
  correct    : episodic memory of (v, pose); top-k match; EMA-update a running offset toward the matched
               earlier pose; confidence trained (BCE) against "is the retrieved pose actually near truth".
"""
import torch, torch.nn as nn, torch.nn.functional as F

KO = 32
class AlloTracker(nn.Module):
    def __init__(self, d=128, store_gap=3, mask_recent=12, temp=0.07, tol=0.6, ema_gain=0.3, topk=3):
        super().__init__()
        self.store_gap, self.mask_recent, self.temp, self.tol, self.d = store_gap, mask_recent, temp, tol, d
        self.ema_gain, self.topk = ema_gain, topk
        self.gru = nn.GRU(KO+4, d, num_layers=2, batch_first=True); self.pose = nn.Linear(d, 2)
        self.view = nn.Sequential(nn.Linear(2*KO, d), nn.GELU(), nn.Linear(d, d))
        self.conf = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.use_memory = True
    def encode_view(self, geo, lm):
        return F.normalize(self.view(torch.cat([geo, lm], -1)), dim=-1)
    def forward(self, geo, lm, act, gt_disp=None, return_aux=False):
        h, _ = self.gru(torch.cat([geo, act], -1)); mu_prior = self.pose(h)
        v = self.encode_view(geo, lm)
        B, T, _ = mu_prior.shape; dev = geo.device
        if not self.use_memory:
            return (mu_prior, v, mu_prior, None, None) if return_aux else mu_prior
        mu_out, keys, vals, times, n_readable = [], [], [], [], 0
        offset = torch.zeros(B, 2, device=dev); ar = torch.arange(B, device=dev); g_list, lbl_list = [], []
        for t in range(T):
            q = v[:, t]
            while n_readable < len(times) and times[n_readable] <= t - self.mask_recent: n_readable += 1
            if n_readable > 0:
                Kmem = torch.stack(keys[:n_readable], 1); Vmem = torch.stack(vals[:n_readable], 1)
                cos = torch.bmm(Kmem, q.unsqueeze(-1)).squeeze(-1); S = cos.shape[1]; k = min(self.topk, S)
                topv, topi = torch.topk(cos, k, dim=1)
                Vtop = torch.gather(Vmem, 1, topi.unsqueeze(-1).expand(-1, -1, 2))
                attn = torch.softmax(cos / self.temp, dim=1)
                retrieved = Vtop.mean(1)                                  # top-k consensus
                spread = Vtop.std(1).mean(-1) if k >= 2 else torch.zeros(B, device=dev)
                gap = (topv[:, 0] - topv[:, 1]) if k >= 2 else torch.zeros(B, device=dev)
                f = torch.stack([topv[:, 0], gap, attn.max(1).values, -spread,
                                 torch.zeros(B, device=dev)], -1)
                g = torch.sigmoid(self.conf(f)).squeeze(-1)
                offset = offset + self.ema_gain * g.unsqueeze(-1) * ((retrieved - mu_prior[:, t]) - offset)
                if gt_disp is not None:
                    label = (torch.norm(retrieved - gt_disp[:, t], dim=-1) < self.tol).float()
                    g_list.append(g.clamp(1e-6, 1-1e-6)); lbl_list.append(label)
            mu_t = mu_prior[:, t] + offset; mu_out.append(mu_t)
            if t % self.store_gap == 0:
                keys.append(v[:, t].detach()); vals.append(mu_t.detach()); times.append(t)
        mu = torch.stack(mu_out, 1)
        if return_aux:
            return mu, v, mu_prior, (torch.stack(g_list) if g_list else None), (torch.stack(lbl_list) if lbl_list else None)
        return mu
