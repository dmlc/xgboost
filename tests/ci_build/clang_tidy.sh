#!/bin/bash

export GTEST_NAME=googletest-release-1.8.1
rm -rf gtest googletest-release-1.7.0 $GTEST_NAME

wget -nc https://github.com/google/googletest/archive/release-1.8.1.zip
unzip -n release-1.8.1.zip
mv $GTEST_NAME gtest && cd gtest
cmake . -DCMAKE_INSTALL_PREFIX=./ins  && make
make install

cd ..
rm -rf release-1.8.1.zip*

python3 tests/ci_build/tidy.py --gtest-path=${PWD}/gtest/ins
