#!/bin/bash
set -e
set -x

make Rpack
mv xgboost/ xgboost_rpack/

mkdir build
cd build
cmake .. -GNinja -DUSE_CUDA=ON -DR_LIB=ON
ninja
cd ..

rm xgboost
rm -v xgboost_rpack/configure
rm -rfv xgboost_rpack/src
mkdir -p xgboost_rpack/src
cp -v lib/xgboost.so xgboost_rpack/src/
echo 'all:' > xgboost_rpack/src/Makefile
echo 'all:' > xgboost_rpack/src/Makefile.win
mv xgboost_rpack/ xgboost/
tar cvzf xgboost.tar.gz xgboost/
