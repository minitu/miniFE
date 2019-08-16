#!/bin/bash

#BSUB -nnodes 8
#BSUB -core_isolation 2
#BSUB -W 60
#BSUB -G asccasc
#BSUB -J minife-n8
#BSUB -q pbatch

date; hostname
echo -n 'Job ID is '; echo $LSB_JOBID
cd /g/g90/choi18/miniFE/cuda/src

for i in 1 2 3 4 5
do
  echo -n 'Run number '; echo $i
  jsrun -n8 -a4 -g4 -c40 -r1 --smpiargs "-mca pml_pami_local_eager_limit 0 -mca pml_pami_remote_eager_limit 0" ./miniFE.x.eval --num_devices 4 -nx 800 -ny 800 -nz 400
done

echo 'Done'
