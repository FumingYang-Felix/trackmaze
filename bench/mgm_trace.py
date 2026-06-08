"""Transparent trace of MGM.step() with tiny dims so every tensor is printable.
Mirrors arch_mgm.py:step() line-for-line, using the REAL model modules (real weights),
printing every intermediate. B=1, d=4, M=4, h=8, K=32 rays.

Trajectory is rigged so step 3 REVISITS step 0 (same obs fed in, motion loops back),
so you can watch the memory read fire a loop closure."""
import torch
from arch_mgm import MGM

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

m = MGM(h=8, d=4, M=4, tau=2.0, use_rot=False, use_mem=True)  # PlainEnc, motion-gated memory
B, K = 1, 32

# obs[t] = (geo[32], lm[32]); reuse obs0 at t=3 to simulate returning to the same place
obs = [(torch.rand(B, K), torch.rand(B, K)) for _ in range(3)]
obs.append(obs[0])                                            # t=3 sees the SAME thing as t=0
# motion[t] in start frame; a square loop that returns near the origin
motion = [torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]),
          torch.tensor([[-1., 0.]]), torch.tensor([[0., -1.]])]

s = m.init_state(B, "cpu")
print("=== MEMORY is pre-allocated FIXED tensors (a ring buffer), not a list ===")
for k in ("mem_p", "mem_mu", "mem_occ"):
    print(f"  s['{k}'].shape = {tuple(s[k].shape)}")
print(f"  s['wptr'] = {s['wptr']}   (write pointer)\n")

tau2 = (m.log_tau.exp() ** 2) + 1e-3
for t in range(4):
    geo_t, lm_t, motion_t = obs[t][0], obs[t][1], motion[t]
    print(f"================= STEP t={t}  (motion={motion_t.tolist()[0]}) =================")

    p = m.enc(geo_t, lm_t)                                    # encoder: (B,64) -> place code (B,4)
    print(f"place code p            = {p}            shape {tuple(p.shape)}")

    s["h"] = m.gru(torch.cat([p, motion_t], -1), s["h"])      # GRU integrates [p, motion]
    mu = s["mu"] + motion_t + m.to_mu(s["h"])                 # dead-reckon + learned residual
    print(f"mu after dead-reckon    = {mu}   (= prev_mu + motion + toMu(h))")

    # ---- motion-gated memory READ ----
    content = (s["mem_p"] * p.unsqueeze(1)).sum(-1) / (m.d ** 0.5)   # (B,M): appearance match per slot
    gatev = -((s["mem_mu"] - mu.unsqueeze(1)) ** 2).sum(-1) / tau2   # (B,M): motion-reachability per slot
    occ_mask = (s["mem_occ"] - 1.0) * 1e4                            # (B,M): -1e4 kills EMPTY slots
    w = torch.softmax(content + gatev + s["mem_imp"] + occ_mask, 1)  # (B,M): attention over slots
    print(f"  mem_occ (filled?)     = {s['mem_occ']}")
    print(f"  content (appearance)  = {content}")
    print(f"  gatev   (reachable?)  = {gatev}")
    print(f"  w = softmax(...)      = {w}   <- which past slot to re-anchor to")

    read_mu = (w.unsqueeze(-1) * s["mem_mu"]).sum(1)                 # (B,2): retrieved position
    read_p = (w.unsqueeze(-1) * s["mem_p"]).sum(1)                   # (B,4): retrieved place code
    conf = torch.sigmoid(m.gate(torch.cat([s["h"], read_p, read_mu - mu], -1)))  # (B,1): trust it?
    mu_corr = mu + conf * (read_mu - mu)                             # loop-closure correction
    print(f"  read_mu               = {read_mu}   conf = {conf.item():.2f}")
    print(f"mu AFTER closure        = {mu_corr}   (= mu + conf*(read_mu - mu))")

    # ---- WRITE current (p, mu) into slot wptr, advance pointer (detached) ----
    wp = s["wptr"]
    s["mem_p"] = s["mem_p"].clone(); s["mem_mu"] = s["mem_mu"].clone(); s["mem_occ"] = s["mem_occ"].clone()
    s["mem_p"][:, wp] = p.detach(); s["mem_mu"][:, wp] = mu_corr.detach(); s["mem_occ"][:, wp] = 1.0
    s["wptr"] = (wp + 1) % m.M
    s["mu"] = mu_corr
    print(f"  WROTE slot {wp}: mem_mu now = {s['mem_mu'].squeeze(0).tolist()}   wptr -> {s['wptr']}\n")
