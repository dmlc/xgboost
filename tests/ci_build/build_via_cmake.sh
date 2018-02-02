#!/usr/bin/env bash

make clean
mkdir build
cd build
cmake .. "$@"
make
