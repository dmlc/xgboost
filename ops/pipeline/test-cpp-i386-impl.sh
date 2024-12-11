#!/bin/bash
## Run C++ tests for i386
## Companion script for ops/pipeline/test-cpp-i386.sh

set -euox pipefail

export CXXFLAGS='-Wno-error=overloaded-virtual -Wno-error=maybe-uninitialized -Wno-error=redundant-move -Wno-narrowing'

mkdir -p build
pushd build

cmake .. \
  -GNinja \
  -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
time ninja -v
# TODO(hcho3): Run gtest for i386
# ./testxgboost

popd
