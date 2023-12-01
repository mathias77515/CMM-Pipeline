#!/bin/bash

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=quiet
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=4      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem=35G
#SBATCH --time=0-10:00:00
#SBATCH --output=mulitple_jobs_%j.log

PYTHON_SCRIPT=main.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

conda activate myqubic

mpirun $interface -np ${SLURM_NTASKS} python ${PYTHON_SCRIPT} $1 $2
