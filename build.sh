#!/bin/bash
# This is a simple script to make xgboost in MAC and Linux
# Basically, it first try to make with OpenMP, if fails, disable OpenMP and make it again.
# This will automatically make xgboost for MAC users who don't have OpenMP support.
# In most cases, type make will give what you want.

# See additional instruction in doc/build.md

#for building static OpenMP lib in MAC for easier installation in MAC
#doesn't work with XCode clang/LLVM since Apple doesn't support, 
#needs brew install gcc 4.9+ with OpenMP. By default the static link is OFF
static_omp=0
if ((${static_omp}==1)); then
    rm libgomp.a
    ln -s `g++ -print-file-name=libgomp.a`
    make clean
    make omp_mac_static=1
    echo "Successfully build multi-thread static link xgboost"
    exit 0
fi

if make; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    make clean
    make no_omp=1
    echo "Successfully build single-thread xgboost"
    echo "If you want multi-threaded version"
    echo "See additional instructions in doc/build.md"
fi
