#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build
cmake -DRABIT_MOCK=ON -DCMAKE_VERBOSE_MAKEFILE=ON ..
make clean
make -j$(nproc)
cd ..
