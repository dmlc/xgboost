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
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
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
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    time ninja -v
    ctest --extra-verbose
    ;;
  cpu-sanitizer)
    echo "--- Build and test XGBoost with sanitizer"
    cmake .. \
      -GNinja \
      -DHIDE_CXX_SYMBOLS=ON \
      -DGOOGLE_TEST=ON \
      -DENABLE_ALL_WARNINGS=ON \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DUSE_SANITIZER=ON \
      -DENABLED_SANITIZERS="address;leak;undefined" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
    time ninja -v
    ./testxgboost --gtest_filter=-*DeathTest*
    ;;
  i386)
    echo "--- Build and test XGBoost for i386 (32-bit)"
    # GCC 16 emits a false-positive stringop-overflow from dmlc::optional.
    export CXXFLAGS="${CXXFLAGS:-} -Wno-error=stringop-overflow"
    cmake .. \
      -GNinja \
      -DGOOGLE_TEST=ON \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
      -DENABLE_ALL_WARNINGS=ON
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
