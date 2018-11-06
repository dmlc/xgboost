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

#pushd xgboost
oldpath=`pwd`
cd ./xgboost/

if echo "${OSTYPE}" | grep -q "darwin"; then
  LIB_XGBOOST=libxgboost.dylib
  # Use OpenMP-capable compiler if possible
  if which g++-5; then
    export CC=gcc-5
    export CXX=g++-5
  elif which g++-7; then
    export CC=gcc-7
    export CXX=g++-7
  elif which g++-8; then
    export CC=gcc-8
    export CXX=g++-8
  elif which clang++; then
    export CC=clang
    export CXX=clang++
  fi
else
  LIB_XGBOOST=libxgboost.so
fi

#remove the pre-compiled .so and trigger the system's on-the-fly compiling
make clean
if make lib/${LIB_XGBOOST} -j4; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    make clean
    make lib/${LIB_XGBOOST} -j4 USE_OPENMP=0
    echo "Successfully build single-thread xgboost"
    echo "If you want multi-threaded version"
    echo "See additional instructions in doc/build.md"
fi
cd $oldpath

set +x
