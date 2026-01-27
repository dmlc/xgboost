#!/bin/bash
## Build XGBoost R package with GPU support and package it in a tarball.
## Users will be able to install it without having CTK installed
## (only a compatible NVIDIA driver is needed).
## This script runs inside the container (via GitHub Actions container support).

set -euox pipefail

python3 ops/script/test_r_package.py --task=pack
mv xgboost/ xgboost_rpack/

mkdir build
cd build
cmake .. -GNinja \
  -DUSE_CUDA=ON \
  -DR_LIB=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache
ninja -v
cd ..

# This super wacky hack is found in cmake/RPackageInstall.cmake.in and
# cmake/RPackageInstallTargetSetup.cmake. This hack lets us bypass the normal build process of R
# and have R use xgboost.so that we've already built.
rm -v xgboost_rpack/configure
rm -rfv xgboost_rpack/src
mkdir -p xgboost_rpack/src
cp -v lib/xgboost.so xgboost_rpack/src/
echo 'all:' > xgboost_rpack/src/Makefile
echo 'all:' > xgboost_rpack/src/Makefile.win
mv xgboost_rpack/ xgboost/
tar cvzf xgboost_r_gpu_linux.tar.gz xgboost/
