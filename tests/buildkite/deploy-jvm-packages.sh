#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  tests/ci_build/ci_build.sh jvm_gpu_build docker \
    --build-arg CUDA_VERSION_ARG=${CUDA_VERSION} \
    --build-arg NCCL_VERSION_ARG=${NCCL_VERSION} \
    tests/ci_build/deploy_jvm_packages.sh ${SPARK_VERSION}
fi
