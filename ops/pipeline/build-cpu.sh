#!/bin/bash
## Build and test XGBoost with CPU

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {cpu,cpu-nonomp,cpu-sanitizer,i386}"
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
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DCMAKE_PREFIX_PATH='/opt/grpc' \
      -DPLUGIN_FEDERATED=ON
    time ninja -v
    echo "--- Run Google Test"
    ctest --extra-verbose
    ;;
  cpu-nonomp)
    echo "--- Build and test XGBoost with OpenMP disabled"
    cmake .. \
      -GNinja \
      -DUSE_OPENMP=OFF \
      -DHIDE_CXX_SYMBOLS=ON \
      -DGOOGLE_TEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
    time ninja -v
    ctest --extra-verbose
    ;;
  cpu-sanitizer)
    echo "--- Build and test XGBoost with sanitizer"
    cmake .. \
      -GNinja \
      -DHIDE_CXX_SYMBOLS=ON \
      -DGOOGLE_TEST=ON \
      -DUSE_DMLC_GTEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DUSE_SANITIZER=ON \
      -DENABLED_SANITIZERS="address;leak;undefined" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
    time ninja -v
    ./testxgboost --gtest_filter=-*DeathTest*
    ;;
  i386)
    echo "--- Build and test XGBoost for i386 (32-bit)"
    export CXXFLAGS='-Wno-error=overloaded-virtual -Wno-error=maybe-uninitialized -Wno-error=redundant-move -Wno-narrowing'
    cmake .. \
      -GNinja \
      -DGOOGLE_TEST=ON \
      -DUSE_DMLC_GTEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
    time ninja -v
    # TODO(hcho3): Run gtest for i386
    # ./testxgboost
    ;;
  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac

popd
