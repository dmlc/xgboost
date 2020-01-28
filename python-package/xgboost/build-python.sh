#!/bin/sh
# This is a simple script to make xgboost in MAC and Linux for python wrapper only
# Basically, it first try to make with OpenMP, if fails, disable OpenMP and make it again.
# This will automatically make xgboost for MAC users who don't have OpenMP support.
# In most cases, type make will give what you want.

# See additional instruction in doc/build.md

# note: this script is build for python package only, and it might have some filename
#       conflict with build.sh which is for everything. 

set -e
set -x

oldpath=`pwd`
cd ./xgboost/

#remove the pre-compiled .so and trigger the system's on-the-fly compiling
mkdir -p build
cd build
if cmake .. && make -j4; then
    echo "Successfully built multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    cmake .. -DUSE_OPENMP=0
    make -j4
    echo "Successfully built single-thread xgboost; training speed may be suboptimal."
    echo "To use all CPU cores for training jobs, install libomp package from Homebrew and re-install XGBoost"
fi
cd $oldpath

set +x
