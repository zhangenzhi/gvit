#!/bin/bash
#SBATCH -A bif146
#SBATCH -J sbcast
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -C nvme

# Move a copy of the env to the NVMe on each node
echo "copying torch_env to each node in the job"
sbcast -pf ./torch_env.tar.gz /mnt/bb/${USER}/torch_env.tar.gz
if [ ! "$?" == "0" ]; then
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
    # your application may pick up partially complete shared library files, which would give you confusing errors.
    echo "SBCAST failed!"
    exit 1
fi