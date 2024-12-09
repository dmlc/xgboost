#!/bin/bash
## Build and test XGBoost with ARM64 CPU
## Companion script for ops/pipeline/build-cpu-arm64.sh

set -euox pipefail

source activate aarch64_test

echo "--- Build libxgboost from the source"
mkdir -p build
pushd build
cmake .. \
  -GNinja \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOLS=ON \
  -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -DBUILD_DEPRECATED_CLI=ON
time ninja -v

echo "--- Run Google Test"
ctest --extra-verbose
popd

echo "--- Build binary wheel"
pushd python-package
rm -rfv dist/*
pip wheel --no-deps -v . --wheel-dir dist/
popd
