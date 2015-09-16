#!/bin/bash
# This is a simple script to make xgboost in MAC and Linux for python wrapper only
# Basically, it first try to make with OpenMP, if fails, disable OpenMP and make it again.
# This will automatically make xgboost for MAC users who don't have OpenMP support.
# In most cases, type make will give what you want.

# See additional instruction in doc/build.md

# note: this script is build for python package only, and it might have some filename
#       conflict with build.sh which is for everything. 


pushd xgboost
if make python; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    make clean
    make python no_omp=1
    echo "Successfully build single-thread xgboost"
    echo "If you want multi-threaded version"
    echo "See additional instructions in doc/build.md"
fi
popd
