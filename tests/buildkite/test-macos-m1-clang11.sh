#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

# Display system info
echo "--- Display system information"
set -x
system_profiler SPSoftwareDataType
sysctl -n machdep.cpu.brand_string
uname -m
set +x

# Create new Conda env
echo "--- Set up Conda env"
. $HOME/mambaforge/etc/profile.d/conda.sh
. $HOME/mambaforge/etc/profile.d/mamba.sh
conda_env=xgboost_dev_$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')
mamba create -y -n ${conda_env} python=3.8
conda activate ${conda_env}
mamba env update -n ${conda_env} --file tests/ci_build/conda_env/macos_cpu_test.yml

# Ensure that XGBoost can be built with Clang 11
echo "--- Build and Test XGBoost with MacOS M1, Clang 11"
set -x
LLVM11_PATH=$(brew --prefix llvm\@11)
mkdir build
pushd build
cmake .. -GNinja -DCMAKE_C_COMPILER=${LLVM11_PATH}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM11_PATH}/bin/clang++ -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON
ninja -v
