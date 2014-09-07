#!/bin/bash
# this is a simple scrip to make xgboost in MAC nad Linux
# basically, it first try to make with OpenMP, if fails, disable OpenMP and make again
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
