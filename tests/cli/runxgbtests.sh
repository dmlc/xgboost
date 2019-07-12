#!/bin/bash

#cleanup enviorment
pkill xgboost
pkill python

#run make in rabit/test to generate librabit_mock
#update config.mk and build xgboost using mock
export DMLC_SUBMIT_CLUSTER=local

submit="python ../../dmlc-core/tracker/dmlc-submit"
# build xgboost with librabit mock
# define max worker retry with dmlc-core local num atempt
# instrument worker failure with mock=xxxx
# check if host recovered from expectected iteration
echo "====== 1. Fault recovery distributed test with checkpointing ======"
$submit --cluster=local --num-workers=10 --local-num-attempt=10 ~/xgboost/xgboost machine.conf 10000 mock=0,10,1,0 mock=1,11,1,0 mock=1,11,1,1 mock=0,11,1,0 mock=4,11,1,0 mock=9,11,1,0 mock=8,11,2,0 mock=4,11,3,0
$submit --cluster=local --num-workers=10 --local-num-attempt=10 ~/xgboost/xgboost machinehist.conf 10000 mock=0,10,1,0 mock=1,11,1,0 mock=1,11,1,1 mock=0,11,1,0 mock=4,11,1,0 mock=9,11,1,0 mock=8,11,2,0 mock=4,11,3,0