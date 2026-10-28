#!/bin/bash
## Build XGBoost with CUDA
## Companion script for ops/pipeline/build-cuda.sh

set -euox pipefail

# PRs and master's cache-warming pass share the same SM75 compiler-cache entries.
cuda_arch_args=()
if [[ -n "${GITHUB_BASE_REF:-}" || "${BRANCH_NAME:-}" == "master" ]]; then
  # PRs reuse the SM75 cache populated by master; other production targets stay covered on master.
  cuda_arch_args=(-DGPU_COMPUTE_VER=75)
fi

cmake_prefix_path=''
if [[ "${USE_RMM:-0}" == 1 ]]
then
  cmake_prefix_path='/opt/rmm;/opt/rmm/lib64/rapids/cmake'
fi

cmake_args=(
  -GNinja
  "-DCMAKE_PREFIX_PATH=${cmake_prefix_path}"
  -DCMAKE_C_COMPILER_LAUNCHER=sccache
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
  -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache
  -DUSE_CUDA=ON
  -DUSE_OPENMP=ON
  -DHIDE_CXX_SYMBOLS=ON
  -DUSE_NCCL=ON
  -DNCCL_INCLUDE_DIR=/usr/include
  -DUSE_DLOPEN_NCCL=ON
  -DGOOGLE_TEST=ON
  -DENABLE_ALL_WARNINGS=ON
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
  "-DCMAKE_CUDA_FLAGS=-Xcompiler=-Wno-error=maybe-uninitialized"
)
if [[ "${USE_RMM:-0}" == 1 ]]; then
  cmake_args+=(-DPLUGIN_RMM=ON)
fi

# Disable CMAKE_COMPILE_WARNING_AS_ERROR option temporarily until
# https://github.com/dmlc/xgboost/issues/10400 is fixed
echo "--- Build libxgboost from the source"
mkdir -p build
pushd build
# CTK 13.3 error:
#   nvcc_internal_extended_lambda_implementation:357:85: error:
#   '*((void*)(&<anonymous>)+8).__nv_hdl_wrapper_t...'
#   may be used uninitialized in this function [-Werror=maybe-uninitialized]
#
# As a result, the speical `-DCMAKE_CUDA_FLAGS` is employed.
cmake .. "${cmake_args[@]}" "${cuda_arch_args[@]}"
time ninja -v

if [[ "${BRANCH_NAME:-}" == "master" ]]; then
  # Populate the SM75 cache for PRs, then rebuild the full production architecture set.
  cmake .. "${cmake_args[@]}" -DGPU_COMPUTE_VER=
  time ninja -v
fi
popd

echo "--- Build binary wheel"
pushd python-package
rm -rfv dist/*
pip wheel --no-deps -v . --wheel-dir dist/
popd
