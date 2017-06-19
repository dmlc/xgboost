#!/bin/bash
# This is a simple script to make xgboost in MAC and Linux
# Basically, it first try to make with OpenMP, if fails, disable OpenMP and make it again.
# This will automatically make xgboost for MAC users who don't have OpenMP support.
# In most cases, type make will give what you want.

# See additional instruction in doc/build.md
set -e

if make; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    make clean_all
    make config=make/minimum.mk
    if [ $? -eq 0 ] ;then
      echo "Successfully build single-thread xgboost"
      echo "If you want multi-threaded version"
      echo "See additional instructions in doc/build.md"
    else
      echo "Failed to build single-thread xgboost"
    fi
fi
