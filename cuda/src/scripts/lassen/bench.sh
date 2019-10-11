#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -J minife-n4

date

cd /g/g90/choi18/miniFE/cuda/src

ranks=4

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -K 2 -r 4 ./miniFE-b -nx 400 -ny 400 -nz 200 --num_devices 1 > miniFE-n"$ranks"-"$iter".out
done
