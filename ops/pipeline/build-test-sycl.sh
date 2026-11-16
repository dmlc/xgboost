#!/bin/bash
## Build and test oneAPI

set -euox pipefail

mkdir build
pushd build
cmake .. -DGOOGLE_TEST=ON -DPLUGIN_SYCL=ON -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -GNinja
ninja
popd

# Both suites use the same SYCL library; build once and package it for Python.
echo "--- Run C++ tests with SYCL"
./build/testxgboost

echo "--- Run Python tests with SYCL"
cd python-package
python --version
pip install -v .
cd ..
pytest -s -v -rxXs --durations=0 ./tests/python-sycl/
