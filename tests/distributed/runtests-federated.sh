#!/bin/bash

set -e

trap "kill 0" EXIT

rm -f ./*.model* ./agaricus* ./*.pem

port=9091
world_size=3

# Generate server and client certificates.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout server-key.pem -out server-cert.pem -subj "/C=US/CN=localhost"
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout client-key.pem -out client-cert.pem -subj "/C=US/CN=localhost"

# Start the federated server.
../../build/plugin/federated/federated_server ${port} ${world_size} server-key.pem server-cert.pem client-cert.pem >/dev/null &

# Split train and test files manually to simulate a federated environment.
split -n l/${world_size} -d ../../demo/data/agaricus.txt.train agaricus.txt.train-
split -n l/${world_size} -d ../../demo/data/agaricus.txt.test agaricus.txt.test-

export FEDERATED_SERVER_ADDRESS="localhost:${port}"
export FEDERATED_WORLD_SIZE=${world_size}
export FEDERATED_SERVER_CERT=server-cert.pem
export FEDERATED_CLIENT_KEY=client-key.pem
export FEDERATED_CLIENT_CERT=client-cert.pem
for ((rank = 0; rank < world_size; rank++)); do
  FEDERATED_RANK=${rank} python test_federated.py &
  pids[${rank}]=$!
done

for pid in ${pids[*]}; do
  wait $pid
done
