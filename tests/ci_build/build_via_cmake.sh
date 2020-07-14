#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build
cmake .. "$@" -GNinja -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DCMAKE_VERBOSE_MAKEFILE=ON
ninja -v
cd ..
