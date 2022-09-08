#!/bin/bash

set -euo pipefail

SPARK_VERSION=3.0.1
CUDA_VERSION=11.0.3

source tests/buildkite/conftest.sh

echo "--- Build XGBoost JVM packages with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

tests/ci_build/ci_build.sh jvm_gpu_build nvidia-docker \
  --build-arg CUDA_VERSION_ARG=${CUDA_VERSION} tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION} -Duse.cuda=ON ${arch_flag}
