#!/bin/bash

set -euo pipefail

echo "--- Build and Test XGBoost with MacOS M1, Clang 11"

source tests/buildkite/conftest.sh

# Ensure that XGBoost can be built with Clang 11
LLVM11_PATH=$(brew --prefix llvm\@11)
mkdir build
pushd build
cmake .. -GNinja -DCMAKE_C_COMPILER=$LLVM11_PATH/bin/clang \
  -DCMAKE_CXX_COMPILER=LLVM11_PATH/bin/clang++ -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON
ninja -v

# Create new Conda env
conda_env=xgboost_dev_$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')
mamba env create -n ${conda_env} --file=tests/ci_build/conda_env/macos_cpu_test.yml
