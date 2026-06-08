#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-10:00
#SBATCH -J navhome
#SBATCH --array=1-12
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/navhome_%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=8
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_navbc.txt)   # reuse: arch seed (4 archs x 3 seeds)
ARCH=$(echo $CFG | awk '{print $1}'); SEED=$(echo $CFG | awk '{print $2}')
echo "task $SLURM_ARRAY_TASK_ID: arch=$ARCH seed=$SEED (maze homing, --pos)"
$P nav_home.py --arch $ARCH --seed $SEED --pos 1 \
  --train_sizes 5 7 --n_train_mazes 1200 --epochs 60 --bs 32 --lr 1e-3 --T_out_mult 14 --eps 0.06 \
  --eval_sizes 5 7 12 20 28 --eval_mazes 20 \
  --tag homepos --out navhome_${ARCH}_s${SEED}.txt
