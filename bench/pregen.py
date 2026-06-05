"""Pre-generate the data caches on the cluster (run once before the SLURM array, to avoid races)."""
from conv_diversity import get_pool
from train_eval import cached, AMB

cached(f"eval_n12_T288_a{AMB}", n=12, n_eps=60, T=288, ambiguity=AMB, seed=7)
for w in [16, 64, 256, 1024]:
    ds = get_pool(w)
    print(f"pool w={w}: {ds['feat'].shape[0]} episodes")
print("pregen done")
