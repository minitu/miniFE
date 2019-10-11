#!/bin/bash

dir="summit"

#for rank in 2
for rank in 4 8 16 32 64 128 256
do
  echo "Running $rank ranks"
  jsrun -n20 -a1 -c1 -K10 -r20 $HOME/work/codes-dumpi/build/src/network-workloads/model-net-mpi-replay --sync=3 --disable_compute=0 --workload_type="dumpi" --workload_file=/ccs/home/jchoi/work/miniFE/cuda/src/dumpi/lassen/n"$rank"- --num_net_traces="$rank" --lp-io-dir="$dir"/n"$rank" -- "$dir"/replay.conf &> "$dir"/n"$rank".out
done
