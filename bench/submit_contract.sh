#!/bin/bash
#SBATCH -J tm_contract
#SBATCH -p lichtman
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-04:00
#SBATCH --array=1-8
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/results/contract-%A_%a.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs_contract.txt)
export OMP_NUM_THREADS=4
echo "task ${SLURM_ARRAY_TASK_ID}: $CFG"
$P contract_tracker.py $CFG --epochs 80
