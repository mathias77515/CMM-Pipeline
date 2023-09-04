#!/bin/bash

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=2        # N
#SBATCH --mem=10G
#SBATCH --time=0-10:00:00
#SBATCH --output=mulitple_jobs_%j.log

PYTHON_SCRIPT=main.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np ${SLURM_NTASKS} python ${PYTHON_SCRIPT} $1 $2

