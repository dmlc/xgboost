#!/bin/bash

trap "kill 0" EXIT

rm -f ./*.model* ./agaricus*

port=9091
world_size=3

../../build/plugin/federated/federated_server ${port} ${world_size} >/dev/null &

# Split train and test files:
split -n l/${world_size} -d ../../demo/data/agaricus.txt.train agaricus.txt.train-
split -n l/${world_size} -d ../../demo/data/agaricus.txt.test agaricus.txt.test-

export FEDERATED_SERVER_ADDRESS="localhost:${port}"
export FEDERATED_WORLD_SIZE=${world_size}
for ((rank = 0; rank < world_size; rank++)); do
  FEDERATED_RANK=${rank} python test_federated.py &
  pids[${rank}]=$!
done

for pid in ${pids[*]}; do
  wait $pid
done
