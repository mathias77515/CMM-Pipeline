#!/bin/bash

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=2        # N
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=0-08:00:00
#SBATCH --output=mulitple_jobs_%j.log
###SBATCH --array=1

SCRATCH_DIRECTORY=jobs/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}

PYTHON_SCRIPT=main.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np ${SLURM_NTASKS} python ${PYTHON_SCRIPT} $1