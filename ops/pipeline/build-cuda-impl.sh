#!/bin/bash
## Build XGBoost with CUDA
## Companion script fro build-cuda.sh

set -euox pipefail

mkdir -p build
pushd build

# Disable CMAKE_COMPILE_WARNING_AS_ERROR option temporarily until
# https://github.com/dmlc/xgboost/issues/10400 is fixed
echo "--- Build libxgboost from the source"
cmake .. \
  -DCMAKE_PREFIX_PATH="/opt/grpc;/workspace/cccl" \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOLS=ON \
  -DPLUGIN_FEDERATED=ON \
  -DUSE_NCCL=ON \
  -DUSE_NCCL_LIB_PATH=ON \
  -DNCCL_INCLUDE_DIR=/usr/include \
  -DUSE_DLOPEN_NCCL=ON \
  -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -DBUILD_DEPRECATED_CLI=ON \
  -GNinja
time ninja -v
popd

echo "--- Build binary wheel"
pushd python-package
rm -rfv dist/*
pip wheel --no-deps -v . --wheel-dir dist/
popd
