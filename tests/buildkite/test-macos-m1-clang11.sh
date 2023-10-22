#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

# Display system info
echo "--- Display system information"
set -x
system_profiler SPSoftwareDataType
sysctl -n machdep.cpu.brand_string
uname -m

# Create new Conda env
. $HOME/mambaforge/etc/profile.d/conda.sh
. $HOME/mambaforge/etc/profile.d/mamba.sh
conda_env=xgboost_dev_$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')
conda activate
mamba env create -n ${conda_env} --file=tests/ci_build/conda_env/macos_cpu_test.yml

# Ensure that XGBoost can be built with Clang 11
echo "--- Build and Test XGBoost with MacOS M1, Clang 11"
LLVM11_PATH=$(brew --prefix llvm\@11)
mkdir build
pushd build
cmake .. -GNinja -DCMAKE_C_COMPILER=${LLVM11_PATH}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM11_PATH}/bin/clang++ -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON
ninja -v
