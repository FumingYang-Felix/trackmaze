#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-06:00
#SBATCH -J demockpt
#SBATCH --array=1-3
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/demockpt_%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=8
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
ARCH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_demockpt.txt)
echo "task $SLURM_ARRAY_TASK_ID: strong ckpt arch=$ARCH"
$P nav_bc.py --arch $ARCH --seed 0 --train_sizes 5 7 --n_train_mazes 3000 --epochs 100 --bs 48 --lr 1e-3 \
  --max_mult 25 --eps 0.08 --eval_sizes 5 7 --eval_mazes 4 --save ckpt_strong_${ARCH}.pt
