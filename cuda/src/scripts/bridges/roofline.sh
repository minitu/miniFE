#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --job-name=rf

set -x

cd /home/jchoi157/gpuroofperf-toolkit/tool

python3 gpuroofperf-cli.py -x "mpiexec -n 2 -print-rank-map" -o miniFE-P2.json /home/jchoi157/miniFE/cuda/src/miniFE.x --num_devices 2 -nx 400 -ny 200 -nz 200
