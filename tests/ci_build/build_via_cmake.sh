#!/usr/bin/env bash
set -e

# Build gtest via cmake
rm -rf gtest
wget https://github.com/google/googletest/archive/release-1.7.0.zip
unzip release-1.7.0.zip
mv googletest-release-1.7.0 gtest && cd gtest
cmake . && make 
mkdir lib && mv libgtest.a lib
cd ..
rm -rf release-1.7.0.zip*

rm -rf build
mkdir build
cd build
cmake .. "$@" -DGOOGLE_TEST=ON -DGTEST_ROOT=../gtest
make clean
make -j
cd ..
