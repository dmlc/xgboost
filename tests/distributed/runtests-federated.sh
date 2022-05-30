#!/bin/bash

set -e

rm -f ./*.model* ./agaricus* ./*.pem

world_size=$(nvidia-smi -L | wc -l)

# Generate server and client certificates.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout server-key.pem -out server-cert.pem -subj "/C=US/CN=localhost"
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout client-key.pem -out client-cert.pem -subj "/C=US/CN=localhost"

# Split train and test files manually to simulate a federated environment.
split -n l/"${world_size}" -d ../../demo/data/agaricus.txt.train agaricus.txt.train-
split -n l/"${world_size}" -d ../../demo/data/agaricus.txt.test agaricus.txt.test-

python test_federated.py "${world_size}"
