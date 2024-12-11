#!/bin/bash
## Build and test XGBoost with AMD64 CPU
## Companion script for ops/pipeline/build-cpu.sh

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {cpu,cpu-sanitizer}"
  exit 1
fi
suite="$1"

mkdir -p build
pushd build

case "${suite}" in
  cpu)
    echo "--- Build libxgboost from the source"
    cmake .. \
      -GNinja \
      -DHIDE_CXX_SYMBOLS=ON \
      -DGOOGLE_TEST=ON \
      -DUSE_DMLC_GTEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DBUILD_DEPRECATED_CLI=ON \
      -DCMAKE_PREFIX_PATH='/opt/grpc' \
      -DPLUGIN_FEDERATED=ON
    time ninja -v
    echo "--- Run Google Test"
    ctest --extra-verbose
    ;;
  cpu-sanitizer)
    echo "--- Run Google Test with sanitizer"
    cmake .. \
      -GNinja \
      -DHIDE_CXX_SYMBOLS=ON \
      -DGOOGLE_TEST=ON \
      -DUSE_DMLC_GTEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DBUILD_DEPRECATED_CLI=ON \
      -DUSE_SANITIZER=ON \
      -DENABLED_SANITIZERS="address;leak;undefined" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
    time ninja -v
    ./testxgboost --gtest_filter=-*DeathTest*
    ;;
  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac

popd
