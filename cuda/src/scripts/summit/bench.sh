#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:30
#BSUB -nnodes 171
#BSUB -J minife-n1024

date

cd $MEMBERWORK/csc357/hpm/apps/miniFE/cuda/src

ranks=1024

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 ./miniFE-b -nx 3200 -ny 1600 -nz 1600 --num_devices 1 > n"$ranks"-"$iter".out
done
