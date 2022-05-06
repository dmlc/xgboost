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

poc -n 2
mkdir -p poc/admin/transfer/hello-xgboost
cp -fr config custom poc/admin/transfer/hello-xgboost
cp server-*.pem client-cert.pem poc/server/
for id in $(eval echo "{1..$world_size}"); do
  cp server-cert.pem client-*.pem poc/site-"$id"/
  cp agaricus.txt.train-site-"$id" poc/site-"$id"/agaricus.txt.train
  cp agaricus.txt.test-site-"$id" poc/site-"$id"/agaricus.txt.test
done
