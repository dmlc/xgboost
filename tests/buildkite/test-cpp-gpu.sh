#!/bin/bash

set -euo pipefail

CUDA_VERSION=11.0.3

source tests/buildkite/conftest.sh

echo "--- Run Google Tests with CUDA, using 4 GPUs"
buildkite-agent artifact download "build/testxgboost" . --step build-cuda
chmod +x build/testxgboost
tests/ci_build/ci_build.sh gpu nvidia-docker \
  --build-arg CUDA_VERSION_ARG=$CUDA_VERSION build/testxgboost

echo "--- Run Google Tests with CUDA, using 4 GPUs, RMM enabled"
rm -rfv build/
buildkite-agent artifact download "build/testxgboost" . --step build-cuda-with-rmm
chmod +x build/testxgboost
tests/ci_build/ci_build.sh rmm nvidia-docker \
  --build-arg CUDA_VERSION_ARG=$CUDA_VERSION bash -c \
  "source activate gpu_test && build/testxgboost --use-rmm-pool"
