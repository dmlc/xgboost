#!/bin/bash

#run make in rabit/test to generate librabit_mock
#update config.mk and build xgboost using mock
export DMLC_SUBMIT_CLUSTER=local

submit="python3 dmlc-core/tracker/dmlc-submit"
# build xgboost with librabit mock
# define max worker retry with dmlc-core local num atempt
# instrument worker failure with mock=xxxx
# check if host recovered from expectected iteration
echo "====== 1. Fault recovery distributed test ======"
exec $submit --cluster=local --num-workers=10 --local-num-attempt=10 $1 $2 mock=0,10,1,0 mock=1,11,1,0 mock=1,11,1,1 mock=0,11,1,0 mock=4,11,1,0 mock=9,11,1,0 mock=8,11,2,0 mock=4,11,3,0 rabit_bootstrap_cache=1 rabit_debug=1
