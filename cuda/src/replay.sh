#!/bin/bash

machine="lassen"
dir="$machine-replay"

#for rank in 256
for rank in 2 4 8 16 32 64 128 256
do
  echo "Running $rank ranks"
  jsrun -n20 -a1 -c1 -K10 -r20 $HPM_PATH/codes/build/src/network-workloads/model-net-mpi-replay --sync=3 --disable_compute=1 --workload_type="dumpi" --workload_file=$HPM_PATH/apps/miniFE/cuda/src/dumpi/lassen/n"$rank"- --num_net_traces="$rank" --lp-io-dir="$dir"/n"$rank" -- $HPM_PATH/conf/$machine/replay.conf &> "$dir"/n"$rank".out
done
