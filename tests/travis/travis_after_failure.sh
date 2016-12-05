#!/bin/bash

if [ ${TASK} == "r_test" ]; then
    cat xgboost/xgboost.Rcheck/*.log
    echo "--------------------------"
    cat xgboost/xgboost.Rcheck/*.out
fi
