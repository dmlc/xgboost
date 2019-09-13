#!/usr/bin/env bash
set -e

# Build gtest via cmake
rm -rf gtest
wget -nc https://github.com/google/googletest/archive/release-1.7.0.zip
unzip -n release-1.7.0.zip
mv googletest-release-1.7.0 gtest && cd gtest
cmake . && make
mkdir lib && mv libgtest.a lib
cd ..
rm -rf release-1.7.0.zip*

rm -rf build
mkdir build
cd build
cmake .. "$@" -DGOOGLE_TEST=ON -DRABIT_MOCK=ON -DGTEST_ROOT=$PWD/../gtest -DCMAKE_VERBOSE_MAKEFILE=ON
make clean
make -j$(nproc)
cd ..
