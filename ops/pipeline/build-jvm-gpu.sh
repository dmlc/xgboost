#!/bin/bash
## Build libxgboost4j.so with CUDA
## This script runs inside the container (via GitHub Actions container support).

set -euox pipefail

source ops/pipeline/classify-git-branch.sh

echo "--- Build libxgboost4j.so with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]; then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

mkdir -p build-gpu
cd build-gpu

cmake .. -GNinja \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=ON \
  -DJVM_BINDINGS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache \
  ${arch_flag}

time ninja -v
