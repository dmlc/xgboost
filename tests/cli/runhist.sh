#!/bin/bash

#run make in rabit/test to generate librabit_mock
#update config.mk and build xgboost using mock
export DMLC_SUBMIT_CLUSTER=local

submit="python ../../dmlc-core/tracker/dmlc-submit"
# build xgboost with librabit mock
# define max worker retry with dmlc-core local num atempt
# instrument worker failure with mock=xxxx
# check if host recovered from expectected iteration
echo "====== 1. Fault recovery distributed test with checkpointing ======"
$submit --cluster=local --num-workers=4 --local-num-attempt=20 ~/xgboost/xgboost machinehist.conf mock=0,8,1,0 mock=1,9,1,0 mock=2,7,1,0 mock=3,8,1,0 mock=0,11,1,0
