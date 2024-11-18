#!/bin/bash
## Build docs for the JVM packages and package it in a tarball
## Note: Note: this script assumes that the user has already built libxgboost4j.so
## and place it in the lib/ directory.

set -euox pipefail

source ops/pipeline/enforce-ci.sh

echo "--- Build JVM packages doc"
python3 ops/docker_run.py \
  --container-id xgb-ci.jvm_gpu_build \
  -- ops/pipeline/build-jvm-doc-impl.sh ${BRANCH_NAME}
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Upload JVM packages doc"
  aws s3 cp jvm-packages/${BRANCH_NAME}.tar.bz2 \
    s3://xgboost-docs/${BRANCH_NAME}.tar.bz2 --acl public-read --no-progress
fi
