#!/bin/bash
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