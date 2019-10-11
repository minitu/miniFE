#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:p100:2
#SBATCH --time=00:10:00
#SBATCH --job-name=miniFE-n8

date

cd /home/jchoi157/miniFE/cuda/src

ranks=8

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

for iter in 1 2 3
do
  echo "Running iteration $iter"
  mpiexec -print-rank-map -n $ranks -ppn 2 -genv I_MPI_DEBUG=5 ./miniFE-b -nx 400 -ny 400 -nz 240 > miniFE-n"$ranks"-"$iter".out
done
