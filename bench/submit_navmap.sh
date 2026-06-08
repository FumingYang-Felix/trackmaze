#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-08:00
#SBATCH -J navmap
#SBATCH --array=1-3
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/navmap_%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=8
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
SEED=$((SLURM_ARRAY_TASK_ID-1))
echo "task $SLURM_ARRAY_TASK_ID: nav_map B seed=$SEED"
$P nav_map.py --seed $SEED --train_sizes 5 7 --n_train_mazes 3000 --epochs 100 --bs 48 --lr 1e-3 \
  --max_mult 25 --eps 0.08 --eval_sizes 5 7 12 20 28 40 --eval_mazes 24 \
  --out navmap_s${SEED}.txt --save ckpt_map_s${SEED}.pt
