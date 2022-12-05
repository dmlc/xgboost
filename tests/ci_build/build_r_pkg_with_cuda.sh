#!/bin/bash
set -e
set -x

if [ "$#" -ne 1 ]
then
  echo "Build the R package tarball with CUDA code. Usage: $0 [commit hash]"
  exit 1
fi

commit_hash="$1"

python tests/ci_build/test_r_package.py --task=pack
mv xgboost/ xgboost_rpack/

mkdir build
cd build
cmake .. -GNinja -DUSE_CUDA=ON -DR_LIB=ON
ninja
cd ..

rm xgboost
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
tar cvzf xgboost_r_gpu_linux_${commit_hash}.tar.gz xgboost/
