#!/bin/bash

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=bigmem
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=6        # N
#SBATCH --mem=40GB
#SBATCH --time=3-00:00:00
#SBATCH --output=multiple_jobs_%j.log
###SBATCH --array=1-100

PYTHON_SCRIPT=main.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRATCH_DIRECTORY=jobs/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}

mpirun $interface -np ${SLURM_NTASKS} python ./${PYTHON_SCRIPT} $1

