#!/bin/bash

set -e

rm -fr ./agaricus* ./*.pem /tmp/nvflare

world_size=2

# Generate server and client certificates.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout server-key.pem -out server-cert.pem -subj "/C=US/CN=localhost"
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout client-key.pem -out client-cert.pem -subj "/C=US/CN=localhost"

# Split train and test files manually to simulate a federated environment.
split -n l/${world_size} --numeric-suffixes=1 -a 1 ../../data/agaricus.txt.train agaricus.txt.train-site-
split -n l/${world_size} --numeric-suffixes=1 -a 1 ../../data/agaricus.txt.test agaricus.txt.test-site-

nvflare poc -n 2 --prepare
mkdir -p /tmp/nvflare/poc/admin/transfer/horizontal-xgboost
cp -fr config custom /tmp/nvflare/poc/admin/transfer/horizontal-xgboost
cp server-*.pem client-cert.pem /tmp/nvflare/poc/server/
for (( site=1; site<=world_size; site++ )); do
  cp server-cert.pem client-*.pem /tmp/nvflare/poc/site-"$site"/
  cp agaricus.txt.train-site-"$site" /tmp/nvflare/poc/site-"$site"/agaricus.txt.train
  cp agaricus.txt.test-site-"$site" /tmp/nvflare/poc/site-"$site"/agaricus.txt.test
done
