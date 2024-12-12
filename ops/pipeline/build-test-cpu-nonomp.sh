#!/bin/bash
## Ensure that XGBoost can function with OpenMP disabled

set -euox pipefail

mkdir -p build
pushd build
cmake .. \
  -GNinja \
  -DUSE_OPENMP=OFF \
  -DHIDE_CXX_SYMBOLS=ON \
  -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -DBUILD_DEPRECATED_CLI=ON
time ninja -v
ctest --extra-verbose
popd
