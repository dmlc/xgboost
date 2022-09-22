#!/bin/bash

set -euo pipefail

SPARK_VERSION=3.0.1
CUDA_VERSION=11.0.3

source tests/buildkite/conftest.sh

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  tests/ci_build/ci_build.sh jvm_gpu_build docker \
    --build-arg CUDA_VERSION_ARG=${CUDA_VERSION} \
    tests/ci_build/deploy_jvm_packages.sh ${SPARK_VERSION}
fi
