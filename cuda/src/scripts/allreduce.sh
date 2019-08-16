#!/bin/bash

#BSUB -nnodes 256
#BSUB -core_isolation 2
#BSUB -W 60
#BSUB -G asccasc
#BSUB -J allreduce-n256
#BSUB -q pbatch

date; hostname
echo -n 'Job ID is '; echo $LSB_JOBID
cd /g/g90/choi18/miniFE/cuda/src

for i in 1 2 3 4 5 6 7 8 9 10
do
  echo -n 'Run number '; echo $i
  jsrun -n1024 -a1 -g1 -c1 -K2 -r4 -d packed ./miniFE.x.eval --num_devices 1 -nx 3200 -ny 1600 -nz 1600
done

echo 'Done'
