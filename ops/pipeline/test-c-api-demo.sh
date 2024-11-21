#!/bin/bash
## Test C API demos

set -euox pipefail

# Build and install XGBoost static library (libxgboost.a)
mkdir build
pushd build
cmake .. -DBUILD_STATIC_LIB=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -GNinja
ninja -v install
popd

# Build and run C API demo with static library
pushd demo/c-api/
mkdir build-c-api-demo
pushd build-c-api-demo
cmake .. -GNinja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
ninja -v
ctest
popd
rm -rf ./build-c-api-demo
popd

# Build and install XGBoost shared library (libxgboost.so)
pushd build
cmake .. -DBUILD_STATIC_LIB=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -GNinja \
  -DPLUGIN_FEDERATED=ON
ninja -v install
popd

# Build and run C API demo with shared library
mkdir demo/c-api/build-c-api-demo
pushd demo/c-api/build-c-api-demo
cmake .. -GNinja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
ninja -v
ctest
popd
./ops/script/verify_link.sh ./demo/c-api/build-c-api-demo/basic/api-demo
./ops/script/verify_link.sh ./demo/c-api/build-c-api-demo/external-memory/external-memory-demo
