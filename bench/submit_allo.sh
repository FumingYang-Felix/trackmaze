#!/bin/bash
#SBATCH -J tm_allo
#SBATCH -p lichtman
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-03:00
#SBATCH --array=1-6
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/results/allo-%A_%a.out
# Stage-1 AlloTracker: allo-canonical place code. canon=true(oracle heading) vs cmd(realistic), 3 seeds.
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_allo.txt)
read CANON S <<< "$CFG"
export OMP_NUM_THREADS=4
echo "task ${SLURM_ARRAY_TASK_ID}: canon=$CANON seed=$S"
$P train_allo.py --canon "$CANON" --seed "$S" --epochs 80
