#!/bin/bash

rm -rf gtest googletest-release-1.7.0
wget -nc https://github.com/google/googletest/archive/release-1.7.0.zip
unzip -n release-1.7.0.zip
mv googletest-release-1.7.0 gtest && cd gtest
cmake . && make
mkdir lib && mv libgtest.a lib
cd ..
rm -rf release-1.7.0.zip*

python3 tests/ci_build/tidy.py --gtest-path=${PWD}/gtest
