#!/bin/bash

set -e

rm -fr ./agaricus* ./*.pem ./poc

world_size=2

# Generate server and client certificates.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout server-key.pem -out server-cert.pem -subj "/C=US/CN=localhost"
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout client-key.pem -out client-cert.pem -subj "/C=US/CN=localhost"

# Split train and test files manually to simulate a federated environment.
split -n l/${world_size} --numeric-suffixes=1 -a 1 ../data/agaricus.txt.train agaricus.txt.train-site-
split -n l/${world_size} --numeric-suffixes=1 -a 1 ../data/agaricus.txt.test agaricus.txt.test-site-

nvflare poc -n 2 --prepare
mkdir -p /tmp/nvflare/poc/admin/transfer/horizontal-xgboost
cp -fr config custom /tmp/nvflare/poc/admin/transfer/horizontal-xgboost
cp server-*.pem client-cert.pem /tmp/nvflare/poc/server/
for id in $(eval echo "{1..$world_size}"); do
  cp server-cert.pem client-*.pem /tmp/nvflare/poc/site-"$id"/
  cp agaricus.txt.train-site-"$id" /tmp/nvflare/poc/site-"$id"/agaricus.txt.train
  cp agaricus.txt.test-site-"$id" /tmp/nvflare/poc/site-"$id"/agaricus.txt.test
done
