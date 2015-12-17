#!/bin/bash

if [ ${TASK} == "R-package" ]; then
    cat xgboost/xgboost.Rcheck/*.log
fi
