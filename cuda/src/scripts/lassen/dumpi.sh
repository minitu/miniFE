#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 64
#BSUB -J minife-d-n256

date

cd /g/g90/choi18/miniFE/cuda/src

ranks=256

echo "Generating DUMPI traces for $ranks ranks"

export LD_LIBRARY_PATH=$HOME/sst-dumpi/install/lib:$LD_LIBRARY_PATH
jsrun -n $ranks -a 1 -c 1 -g 1 -K 2 -r 4 ./miniFE-d -nx 1600 -ny 1600 -nz 800 --num_devices 1 > miniFE-d-n"$ranks".out
