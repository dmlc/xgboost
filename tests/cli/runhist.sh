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
$submit --cluster=local --num-workers=10 --local-num-attempt=1 ~/xgboost/xgboost machinehist.conf mock=0,7,1,0 | exit 0
echo "====== 1. Fault recovery distributed test with allreduce ======"
$submit --cluster=local --num-workers=10 --local-num-attempt=1 ~/xgboost/xgboost machinehist.conf mock=0,10,1,0 | exit 0
