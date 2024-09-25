#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

# Display system info
echo "--- Display system information"
set -x
system_profiler SPSoftwareDataType
sysctl -n machdep.cpu.brand_string
uname -m
set +x

# Ensure that XGBoost can be built with Clang 11
echo "--- Build and Test XGBoost with MacOS M1, Clang 11"
set -x
LLVM11_PATH=$(brew --prefix llvm\@11)
mkdir build
pushd build
cmake .. -GNinja -DCMAKE_C_COMPILER=${LLVM11_PATH}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM11_PATH}/bin/clang++ -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON
ninja -v
./testxgboost
