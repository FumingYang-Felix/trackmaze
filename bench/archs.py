"""The baseline architecture zoo for the TrackMaze localization task.
All map an egocentric feature stream (B,T,din) -> per-step egocentric displacement (B,T,2).

  GRUTracker / LSTMTracker  - gated recurrence (strong integrator; the GRU already failed to use
                              landmarks for loop-closure at test time -> our floor).
  TransformerTracker        - causal self-attention (attention IS a matching/retrieval op, so this is
                              the natural loop-closure mechanism; sinusoidal PE so it can run on
                              OOD-longer sequences). Weak at long-horizon sequential state (TC0).
  SSMTracker                - a minimal *selective* diagonal state-space block h=a*h+(1-a)*Bx with
                              input-dependent decay a (Mamba's core recurrence), stacked w/ residuals.
"""
import math
import torch, torch.nn as nn

class GRUTracker(nn.Module):
    def __init__(self, din, h=128, layers=2):
        super().__init__(); self.rnn = nn.GRU(din, h, layers, batch_first=True); self.head = nn.Linear(h, 2)
    def forward(self, x): o,_ = self.rnn(x); return self.head(o)

class LSTMTracker(nn.Module):
    def __init__(self, din, h=128, layers=2):
        super().__init__(); self.rnn = nn.LSTM(din, h, layers, batch_first=True); self.head = nn.Linear(h, 2)
    def forward(self, x): o,_ = self.rnn(x); return self.head(o)

class TransformerTracker(nn.Module):
    def __init__(self, din, d=128, layers=4, heads=4, ff=256):
        super().__init__()
        self.inp = nn.Linear(din, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers); self.head = nn.Linear(d, 2); self.d = d
    def _pe(self, T, dev):                       # sinusoidal positional encoding (extrapolates to OOD lengths)
        pos = torch.arange(T, device=dev).float()[:, None]
        i = torch.arange(self.d, device=dev).float()[None, :]
        ang = pos / (10000 ** (2 * (i // 2) / self.d))
        pe = torch.zeros(T, self.d, device=dev); pe[:, 0::2] = ang[:, 0::2].sin(); pe[:, 1::2] = ang[:, 1::2].cos()
        return pe
    def forward(self, x):
        B, T, _ = x.shape; h = self.inp(x) + self._pe(T, x.device)[None]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)   # causal
        return self.head(self.enc(h, mask=mask))

class SSMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a_proj = nn.Linear(d, d); self.b_proj = nn.Linear(d, d); self.out = nn.Linear(d, d); self.norm = nn.LayerNorm(d)
    def forward(self, x):                        # (B,T,d) selective leaky integrator, sequential scan
        z = self.norm(x); a = torch.sigmoid(self.a_proj(z) + 2.0); bx = self.b_proj(z)   # bias->slow decay by default
        B, T, d = z.shape; h = torch.zeros(B, d, device=x.device, dtype=x.dtype); ys = []
        for t in range(T):
            h = a[:, t] * h + (1 - a[:, t]) * bx[:, t]; ys.append(h)
        return x + self.out(torch.stack(ys, 1))

class SSMTracker(nn.Module):
    def __init__(self, din, d=128, layers=2):
        super().__init__(); self.inp = nn.Linear(din, d)
        self.blocks = nn.ModuleList(SSMBlock(d) for _ in range(layers)); self.head = nn.Linear(d, 2)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.head(h)

from arch_kalman_assoc import KalmanAssocTracker   # learned Kalman predict-correct + vocab-independent assoc

REGISTRY = {"gru": GRUTracker, "lstm": LSTMTracker, "transformer": TransformerTracker, "ssm": SSMTracker,
            "kanlc": KalmanAssocTracker}

def build(arch, din): return REGISTRY[arch](din)
