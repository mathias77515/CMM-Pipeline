#!/bin/bash -l

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=bigmem
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=2      # n
#SBATCH --cpus-per-task=6        # N
#SBATCH --mem=40GB
#SBATCH --time=3-00:00:00
#SBATCH --output=multiple_jobs_%j.log
###SBATCH --array=1-100

PYTHON_SCRIPT=main.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRATCH_DIRECTORY=jobs/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}

if [[ $(hostname) == "node01" || $(hostname) == "node02" || $(hostname) == "node03" || $(hostname) == "node04" || $(hostname) == "node05" 
                              || $(hostname) == "node06" || $(hostname) == "node07" || $(hostname) == "node08" || $(hostname) == "node09" 
                              || $(hostname) == "node10" || $(hostname) == "node11" || $(hostname) == "node12" || $(hostname) == "node13" 
                              || $(hostname) == "node14" || $(hostname) == "node15" || $(hostname) == "node16" ]]; then
    
    interface="--mca btl_tcp_if_include enp24s0f0np0"
else
    interface="-mca btl_tcp_if_include enp24s0f1"
fi

eval "$(/soft/anaconda3/bin/conda shell.bash hook)"
conda activate venv-qubic

mpirun $interface -np ${SLURM_NTASKS} python ./${PYTHON_SCRIPT} $1

