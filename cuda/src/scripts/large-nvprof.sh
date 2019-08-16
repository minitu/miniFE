#!/bin/bash

#BSUB -nnodes 64
#BSUB -core_isolation 2
#BSUB -W 60
#BSUB -G asccasc
#BSUB -J minife-nv-n64
#BSUB -q pbatch

date; hostname
echo -n 'Job ID is '; echo $LSB_JOBID
cd /g/g90/choi18/miniFE/cuda/src

jsrun -n64 -a4 -g4 -c40 -r1 --smpiargs "-mca pml_pami_local_eager_limit 0 -mca pml_pami_remote_eager_limit 0" nvprof -u ms --profile-from-start off --log-file simple-N64-%q{OMPI_COMM_WORLD_RANK}.nvprof ./miniFE.x.eval --num_devices 4 -nx 1600 -ny 1600 -nz 800

echo 'Done'
