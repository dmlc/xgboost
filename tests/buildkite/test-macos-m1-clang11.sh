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

# Build XGBoost4J binary
echo "--- Build libxgboost4j.dylib"
set -x
mkdir build
pushd build
export JAVA_HOME=$(/usr/libexec/java_home)
cmake .. -GNinja -DJVM_BINDINGS=ON -DUSE_OPENMP=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15
ninja -v
popd
rm -rf build
otool -L lib/libxgboost.dylib
set +x

echo "--- Upload Python wheel"
set -x
pushd lib
mv -v libxgboost4j.dylib libxgboost4j_m1_${BUILDKITE_COMMIT}.dylib
buildkite-agent artifact upload libxgboost4j_m1_${BUILDKITE_COMMIT}.dylib
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp libxgboost4j_m1_${BUILDKITE_COMMIT}.dylib \
    s3://xgboost-nightly-builds/${BRANCH_NAME}/libxgboost4j/ \
    --acl public-read --no-progress
fi
popd
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
