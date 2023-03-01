#!/bin/bash
set -e
set -x

if [ "$#" -ne 1 ]
then
  echo "Build the R package tarball with CUDA code. Usage: $0 [commit hash]"
  exit 1
fi

commit_hash="$1"
# Clear all positional args
set --

source activate
python tests/ci_build/test_r_package.py --task=pack
mv xgboost/ xgboost_rpack/

mkdir build
cd build
cmake .. -G"Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON -DR_LIB=ON -DLIBR_HOME="c:\\Program Files\\R\\R-3.6.3"
cmake --build . --config Release --parallel
cd ..

rm xgboost
# This super wacky hack is found in cmake/RPackageInstall.cmake.in and
# cmake/RPackageInstallTargetSetup.cmake. This hack lets us bypass the normal build process of R
# and have R use xgboost.dll that we've already built.
rm -v xgboost_rpack/configure
rm -rfv xgboost_rpack/src
mkdir -p xgboost_rpack/src
cp -v lib/xgboost.dll xgboost_rpack/src/
echo 'all:' > xgboost_rpack/src/Makefile
echo 'all:' > xgboost_rpack/src/Makefile.win
mv xgboost_rpack/ xgboost/
/c/Rtools/bin/tar -cvf xgboost_r_gpu_win64_${commit_hash}.tar xgboost/
/c/Rtools/bin/gzip -9c xgboost_r_gpu_win64_${commit_hash}.tar > xgboost_r_gpu_win64_${commit_hash}.tar.gz
