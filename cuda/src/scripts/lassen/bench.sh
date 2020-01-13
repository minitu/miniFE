#!/bin/bash
#BSUB -G asccasc
#BSUB -W 30
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 256
#BSUB -J minife-n1024

date

cd /g/g90/choi18/hpm/apps/miniFE/cuda/src

ranks=1024

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -K 2 -r 4 ./miniFE-b -nx 3200 -ny 1600 -nz 1600 --num_devices 1 > n"$ranks"-"$iter".out
done
