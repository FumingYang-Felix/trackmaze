"""configs_brute2.txt: (1) confirm the high-upside knobs (loop=0.5, trainN=8, combo) with 5 seeds;
(2) wave-2 revisit-consistency loss sweep (the structural fix: same true cell -> same latent)."""
def cfg(canon="cmd", loop=0.0, rot=0.09, tscale=1.0, trainN=6, ema=0.3, gap=3, recent=12, topk=3, cons=0.0, seed=0):
    return (f"--canon {canon} --loop {loop} --rot_noise {rot} --tscale {tscale} --train_n {trainN} "
            f"--ema_gain {ema} --store_gap {gap} --mask_recent {recent} --topk {topk} --consistency {cons} --seed {seed}")

out = []
def add(seeds, **kw):
    for s in seeds: out.append(cfg(seed=s, **kw))

S5, S3 = (0, 1, 2, 3, 4), (0, 1, 2)
# (1) confirm high-upside knobs, 5 seeds
add(S5, loop=0.5)
add(S5, trainN=8)
add(S5, loop=0.5, trainN=8)
# (2) wave-2: revisit-consistency loss
add(S3, cons=0.5)
add(S3, cons=2.0)
add(S3, cons=2.0, trainN=8)
add(S3, cons=2.0, loop=0.5)

open("configs_brute2.txt", "w").write("\n".join(out) + "\n")
print(f"{len(out)} configs")
