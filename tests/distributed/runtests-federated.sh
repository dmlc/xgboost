#!/bin/bash

trap "kill 0" EXIT

rm -f ./*.model*

port=9091
world_size=3

../../build/plugin/federated/federated_server ${port} ${world_size} &

export FEDERATED_SERVER_ADDRESS="localhost:${port}"
export FEDERATED_WORLD_SIZE=${world_size}
for ((rank = 0; rank < world_size; rank++)); do
  FEDERATED_RANK=${rank} python test_basic.py &
  pids[${rank}]=$!
done

for pid in ${pids[*]}; do
  wait $pid
done
