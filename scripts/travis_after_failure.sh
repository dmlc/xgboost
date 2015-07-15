#!/bin/bash

if [ ${TASK} == "R-package" ]; then
    cat R-package/xgboost.Rcheck/*.log
fi
