#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

# Work around https://github.com/dmlc/xgboost/issues/11154
# Allocate extra space in /dev/shm to enable NCCL
export CI_DOCKER_EXTRA_PARAMS_INIT='-e NCCL_RAS_ENABLE=0 --shm-size=4g'

echo "--- Run Google Tests with CUDA, using multiple GPUs"
buildkite-agent artifact download "build/testxgboost" . --step build-cuda
chmod +x build/testxgboost
tests/ci_build/ci_build.sh gpu --use-gpus \
  --build-arg CUDA_VERSION_ARG=$CUDA_VERSION \
  --build-arg RAPIDS_VERSION_ARG=$RAPIDS_VERSION \
  --build-arg NCCL_VERSION_ARG=$NCCL_VERSION \
  build/testxgboost --gtest_filter=*MGPU*
