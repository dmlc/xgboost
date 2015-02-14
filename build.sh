#!/bin/bash
# This is a simple script to make xgboost in MAC and Linux
# Basically, it first try to make with OpenMP, if fails, disable OpenMP and make it again.
# This will automatically make xgboost for MAC users who don't have OpenMP support.
# In most cases, type make will give what you want.

# download rabit

if make; then
    echo "Successfully build multi-thread xgboost"
else
    echo "-----------------------------"
    echo "Building multi-thread xgboost failed"
    echo "Start to build single-thread xgboost"
    make clean
    make no_omp=1
    echo "Successfully build single-thread xgboost"
fi
