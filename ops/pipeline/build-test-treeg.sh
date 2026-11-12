#!/bin/bash
## Build and test the Tree-G plugin

set -euox pipefail

# The Tree-G sources are compiled into `objxgboost', so both the C++ test binary
# and the Python extension come from a single build.
mkdir build
pushd build
cmake .. -GNinja -DGOOGLE_TEST=ON -DPLUGIN_TREEG=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
ninja
popd

echo "--- Run C++ tests with Tree-G"
./build/testxgboost --gtest_filter='*TreeG*'

echo "--- Run Python tests with Tree-G"
cd python-package
python --version
pip install -v .
cd ..
pytest -s -v -rxXs --durations=0 ./tests/python/test_treeg.py
