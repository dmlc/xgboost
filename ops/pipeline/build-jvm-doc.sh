#!/bin/bash
## Build docs for the JVM packages and package it in a tarball
## Note: this script assumes that the user has already built libxgboost4j.so
## and place it in the lib/ directory.

set -euox pipefail

echo "--- Build JVM packages doc"

if [[ -z ${BRANCH_NAME:-} ]]
then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

if [[ ! -f lib/libxgboost4j.so ]]
then
  echo "Must place libxgboost4j.so in lib/ first"
  exit 2
fi

python3 ops/docker_run.py \
  --container-id xgb-ci.jvm_gpu_build \
  -- ops/pipeline/build-jvm-doc-impl.sh ${BRANCH_NAME}
