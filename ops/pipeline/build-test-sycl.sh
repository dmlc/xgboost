#!/bin/bash
## Build and test oneAPI

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {gtest,pytest}"
  exit 1
fi

suite="$1"

mkdir build
pushd build
cmake .. -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DPLUGIN_SYCL=ON -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -GNinja
ninja
popd

case "$suite" in
  gtest)
    ./build/testxgboost
    ;;
  pytest)
    cd python-package
    python --version
    pip install -v .
    cd ..
    pytest -s -v -rxXs --durations=0 ./tests/python-sycl/
    ;;
esac
