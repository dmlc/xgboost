#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

# Allocate extra space in /dev/shm to enable NCCL
export CI_DOCKER_EXTRA_PARAMS_INIT='--shm-size=4g'

echo "--- Run Google Tests with CUDA, using multiple GPUs"
buildkite-agent artifact download "build/testxgboost" . --step build-cuda
chmod +x build/testxgboost
tests/ci_build/ci_build.sh gpu nvidia-docker \
  --build-arg CUDA_VERSION_ARG=$CUDA_VERSION \
  --build-arg RAPIDS_VERSION_ARG=$RAPIDS_VERSION \
  build/testxgboost --gtest_filter=*MGPU*
