#!/bin/bash

set -e

trap "kill 0" EXIT

rm -f ./*.model* ./agaricus* ./*.pem ./.*.srl

port=9091
world_size=3

# Generate server and client certificates.
# 1. Generate CA's private key and self-signed certificate.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout ca-key.pem -out ca-cert.pem -subj "/C=US/CN=localhost"

# 2. Generate gRPC server's private key and certificate signing request (CSR).
openssl req -newkey rsa:2048 -nodes -keyout server-key.pem -out server-req.pem -subj "/C=US/CN=localhost"

# 3. Use CA's private key to sign gRPC server's CSR and get back the signed certificate.
openssl x509 -req -in server-req.pem -days 7 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem

# 4. Generate client's private key and certificate signing request (CSR).
openssl req -newkey rsa:2048 -nodes -keyout "client-key.pem" -out "client-req.pem" -subj "/C=US/CN=localhost"

# 5. Use CA's private key to sign client's CSR and get back the signed certificate.
openssl x509 -req -in "client-req.pem" -days 7 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out "client-cert.pem"

# Start the federated server.
../../build/plugin/federated/federated_server ${port} ${world_size} client-cert.pem server-key.pem server-cert.pem >/dev/null &

# Split train and test files manually to simulate a federated environment.
split -n l/${world_size} -d ../../demo/data/agaricus.txt.train agaricus.txt.train-
split -n l/${world_size} -d ../../demo/data/agaricus.txt.test agaricus.txt.test-

export FEDERATED_SERVER_ADDRESS="localhost:${port}"
export FEDERATED_WORLD_SIZE=${world_size}
export FEDERATED_CA_CERT=server-cert.pem
export FEDERATED_CLIENT_KEY=client-key.pem
export FEDERATED_CLIENT_CERT=client-cert.pem
for ((rank = 0; rank < world_size; rank++)); do
  FEDERATED_RANK=${rank} python test_federated.py &
  pids[${rank}]=$!
done

for pid in ${pids[*]}; do
  wait $pid
done
