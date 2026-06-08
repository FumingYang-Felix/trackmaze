#!/bin/bash
#SBATCH -p sapphire
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 0-08:00
#SBATCH -J scal
#SBATCH -o /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench/deployable/scaling_npc40_%j.out
cd /n/netscratch/lichtman_lab/Lab/fumingyang/trackmaze/bench
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
P=/n/home06/fumingyang/.conda/envs/asr_env/bin/python
$P deployable/sweep_scaling.py --sizes 8 16 24 32 44 64 --mazes 5 --npc 40 --workers 28 --loop 0.9
