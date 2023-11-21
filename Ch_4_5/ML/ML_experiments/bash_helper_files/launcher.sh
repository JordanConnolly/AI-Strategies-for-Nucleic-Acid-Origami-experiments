#!/bin/bash -l

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Classical "module load" in the main script
# module load intel
# module load Python/3.8.2-GCCcore-9.3.0
# Run Script
# source /mnt/nfs/home/user/miniconda3
# source /mnt/nfs/home/user/miniconda3/etc/profile.d/conda.sh
activate base

# Start with python input script
INPUTFILE=$(pwd)/extra_trees_RFE_30.py
python $INPUTFILE

echo Finishing Job
