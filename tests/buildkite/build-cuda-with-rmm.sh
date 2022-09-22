#!/bin/bash

set -euo pipefail

CUDA_VERSION=11.0.3

source tests/buildkite/conftest.sh

echo "--- Build with CUDA ${CUDA_VERSION}, RMM enabled"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

command_wrapper="tests/ci_build/ci_build.sh rmm docker --build-arg "`
                `"CUDA_VERSION_ARG=$CUDA_VERSION"

echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/build_via_cmake.sh --conda-env=gpu_test -DUSE_CUDA=ON \
  -DUSE_NCCL=ON -DPLUGIN_RMM=ON -DBUILD_WITH_CUDA_CUB=ON ${arch_flag}

echo "-- Stash C++ test executable (testxgboost)"
buildkite-agent artifact upload build/testxgboost
