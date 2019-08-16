#!/bin/bash

#BSUB -nnodes 4
#BSUB -core_isolation 2
#BSUB -W 240
#BSUB -G asccasc
#BSUB -J rf-g16-notb
#BSUB -q pbatch

date; hostname
echo -n 'Job ID is '; echo $LSB_JOBID
cd /g/g90/choi18/gpuroofperf-toolkit/tool

python3 gpuroofperf-cli.py -x "jsrun -n4 -a4 -g4 -c40 -r1" -o miniFE-P16.json /g/g90/choi18/miniFE/cuda/src/miniFE.x --num_devices 4 -nx 800 -ny 400 -nz 400

echo 'Done'
