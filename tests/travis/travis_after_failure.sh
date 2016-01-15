#!/bin/bash

if [ ${TASK} == "r_test" ]; then
    cat xgboost/xgboost.Rcheck/*.log
fi
