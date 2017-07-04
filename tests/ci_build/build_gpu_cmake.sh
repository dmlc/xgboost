#!/usr/bin/env bash

mkdir build
cd build
cmake .. -DPLUGIN_UPDATER_GPU=ON
make
