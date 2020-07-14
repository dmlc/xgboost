#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DRABIT_MOCK=ON -DCMAKE_VERBOSE_MAKEFILE=ON ..
ninja -v
cd ..
