#!/bin/bash
#SBATCH -J tm_exp3
#SBATCH -p lichtman
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-04:00
#SBATCH --array=1-12
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/results/exp3-%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_exp3.txt)
export OMP_NUM_THREADS=4
echo "task ${SLURM_ARRAY_TASK_ID}: $CFG"
$P train_allo.py $CFG --epochs 80
