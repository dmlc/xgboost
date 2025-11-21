#!/bin/bash
## Ensure that XGBoost can function with OpenMP disabled

set -euox pipefail

conda env list

echo $CONDA_PREFIX
echo $PATH
echo $(which cmake)

conda activate cpp_test

mkdir -p build
pushd build
cmake .. \
  -GNinja \
  -DUSE_OPENMP=OFF \
  -DHIDE_CXX_SYMBOLS=ON \
  -DGOOGLE_TEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
time ninja -v
ctest --extra-verbose
popd
