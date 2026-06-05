"""Emit configs_brute.txt: a structured brute-force sweep (vary one lever at a time around a baseline +
key regime combos), 2 seeds each. Covers A (loop/noise/tscale/trainN), B (ema/topk), C (canon),
E (store_gap/mask_recent). Heavy architectural levers (pose-graph, grid latent, occupancy map) are wave-2."""
def cfg(canon="cmd", loop=0.0, rot=0.09, tscale=1.0, trainN=6, ema=0.3, gap=3, recent=12, topk=3, seed=0):
    return (f"--canon {canon} --loop {loop} --rot_noise {rot} --tscale {tscale} --train_n {trainN} "
            f"--ema_gain {ema} --store_gap {gap} --mask_recent {recent} --topk {topk} --seed {seed}")

out = []
def add(seeds=(0, 1), **kw):
    for s in seeds: out.append(cfg(seed=s, **kw))

add()                                             # baseline (canon=cmd)
for L in (0.1, 0.25, 0.5): add(loop=L)            # A: loopy maze (top suspect)
for R in (0.15, 0.25): add(rot=R)                 # A: bigger drift
add(tscale=2.0)                                   # A: longer episodes
add(canon="true")                                 # C: oracle heading (upper bound)
for E in (0.1, 0.6, 1.0): add(ema=E)              # B: correction gain
add(gap=6); add(recent=24)                        # E: memory cadence / recency
for K in (1, 5): add(topk=K)                      # B: retrieval breadth
add(trainN=8)                                     # F: bigger train maze
add(loop=0.25, rot=0.15)                          # combos: regime
add(loop=0.25, tscale=2.0)
add(loop=0.25, ema=0.6)
add(loop=0.5, rot=0.15)

open("configs_brute.txt", "w").write("\n".join(out) + "\n")
print(f"{len(out)} configs written")
