#!/bin/bash
#SBATCH -J tm_rlc
#SBATCH -p lichtman
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-02:00
#SBATCH --array=1-6
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/results/rlc-%A_%a.out
# Stage-1 structured correction: RELATIONAL (permutation-invariant) vs ABSOLUTE-id place key, 3 seeds.
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_rlc.txt)
read REL S <<< "$CFG"
export OMP_NUM_THREADS=4
FLAG=""; [ "$REL" = "1" ] && FLAG="--relational"
echo "task ${SLURM_ARRAY_TASK_ID}: relational=$REL seed=$S"
$P train_rlc.py --retr topk --update ema --feats rich --seed "$S" --epochs 80 $FLAG
