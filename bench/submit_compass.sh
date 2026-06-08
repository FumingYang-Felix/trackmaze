#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-10:00
#SBATCH -J navcomp
#SBATCH --array=1-3
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/navcomp_%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=8
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_compass.txt)
ARCH=$(echo $CFG | awk '{print $1}'); SEED=$(echo $CFG | awk '{print $2}')
echo "task $SLURM_ARRAY_TASK_ID: arch=$ARCH seed=$SEED (compass / strong)"
$P nav_bc.py --arch $ARCH --seed $SEED \
  --train_sizes 5 7 --n_train_mazes 3000 --epochs 100 --bs 48 --lr 1e-3 --max_mult 25 --eps 0.08 \
  --eval_sizes 5 7 12 20 28 40 --eval_mazes 24 \
  --tag strong --out navbc_strong_${ARCH}_s${SEED}.txt
