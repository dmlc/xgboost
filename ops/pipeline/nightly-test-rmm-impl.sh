#!/bin/bash
## Companion script for ops/pipeline/nightly-test-rmm.sh

set -eo pipefail
# Cannot set -u before Conda env activation

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [rmm_version]"
  exit 1
fi
rmm_version="$1"

# Set up Conda env
gosu root chown -R $(id -u):$(id -g) /opt/miniforge/envs /opt/miniforge/pkgs/cache
gosu root chown $(id -u):$(id -g) /opt/miniforge/pkgs
mamba create -y -n rmm_test -c conda-forge -c rapidsai-nightly python=3.13 \
  cuda-version=13.0 cxx-compiler cuda-cudart-dev cuda-nvcc gcc_linux-64=14.* ninja \
  gtest nccl "rmm=${rmm_version%.*}.*,>=0.0.0a0"

source activate rmm_test

if [[ "${BUILD_ONLY_SM75:-}" == 1 ]]
then
  cmake_args='-DGPU_COMPUTE_VER=75'
else
  cmake_args=''
fi

cmake_args="${cmake_args} -DPLUGIN_RMM=ON"

mkdir -p build
cd build
cmake .. \
  -GNinja \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOLS=ON \
  -DUSE_NCCL=ON \
  -DUSE_DLOPEN_NCCL=ON \
  -DGOOGLE_TEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  ${cmake_args}
ninja -v
