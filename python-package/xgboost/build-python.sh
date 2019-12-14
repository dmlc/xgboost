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

GCC_PATH=""
if echo "${OSTYPE}" | grep -q "darwin"; then
  # Use OpenMP-capable compiler if possible
  if [ "v"`sw_vers -buildVersion` -ge "v17G65" ]; then
    if which g++-9; then
      GCC_PATH="CC=gcc-9 CXX=g++-9"
    else
      echo "For MacOS version higher than High Sierra, please install gcc@9 first."
    fi
  elif which g++-5; then
    GCC_PATH="CC=gcc-5 CXX=g++-5"
  elif which g++-7; then
    GCC_PATH="CC=gcc-7 CXX=g++-7"
  elif which g++-8; then
    GCC_PATH="CC=gcc-8 CXX=g++-8"
  elif which clang++; then
    GCC_PATH="CC=clang CXX=clang++"
  fi
fi

#remove the pre-compiled .so and trigger the system's on-the-fly compiling
mkdir -p build
cd build
if [ -f * ]; then
  rm -r *
fi
if eval $GCC_PATH" cmake .." && eval $GCC_PATH" make -j4"; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    eval $GCC_PATH" cmake .. -DUSE_OPENMP=0"
    eval $GCC_PATH" make -j4"
    echo "Successfully build single-thread xgboost"
    echo "If you want multi-threaded version"
    echo "See additional instructions in doc/build.md"
fi
cd $oldpath

set +x
