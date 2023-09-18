# Components Map-Making

Repository that contain End-2-End pipeline for components reconstruction directly from Time-Ordered Data (TOD) for QUBIC experiments.

# Description



# Run 

The code can be run locally but much more efficient in Computing Cluster using SLURM system. To send jobs on computing clusters with SLURM system, use the command :

```
sbatch main.sh {SEED_CMB} {SEED_NOISE}
```

To modify memory requirements, please modify `main.sh` file, especially lines :

* `#SBATCH --partition=your_partition` to run on different sub-systems.
* `#SBATCH --nodes=number_of_nodes` to split data on different nodes.
* `#SBATCH --ntasks-per-node=Number_of_taks` to run several MPI tasks.
* `#SBATCH --cpus-per-task=number_of_CPU` to ask for several CPU for OpenMP system.
* `#SBATCH --mem=number_of_Giga` to ask for memory (in GB, let the letter G at the end i.e 6G)
* `#SBATCH --time=day-hours:minutes:seconds` to ask for more execution time.
