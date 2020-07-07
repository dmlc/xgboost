#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build
cmake .. "$@" -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DCMAKE_VERBOSE_MAKEFILE=ON -GNinja
ninja clean
/usr/bin/time -v ninja -v
cd ..
