#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 86
#BSUB -J minife-n512

date

cd $MEMBERWORK/csc357/hpm/apps/miniFE/cuda/src

ranks=512

jsrun -n $ranks -a 1 -c 1 -g 1 ./miniFE.x -nx 1600 -ny 1600 -nz 1600 --num_devices 1 &> n"$ranks"-debug.out
