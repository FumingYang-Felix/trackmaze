#!/bin/bash
#SBATCH -J tm_div
#SBATCH -p lichtman
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-00:40
#SBATCH --array=1-24
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/results/slurm-%A_%a.out
# Convergent diversity x capacity sweep (4 worlds x 2 caps x 3 seeds), 120 epochs, one task per config.
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs.txt)
read W C S <<< "$CFG"
export OMP_NUM_THREADS=4
echo "task ${SLURM_ARRAY_TASK_ID}: worlds=$W cap=$C seed=$S"
$P conv_diversity.py --worlds "$W" --cap "$C" --seed "$S" --epochs 120
