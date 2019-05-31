#!/bin/bash

#run make in rabit/test to generate librabit_mock
#update config.mk and build xgboost using mock
rm -f *.model*

export DMLC_SUBMIT_CLUSTER=local

submit="timeout 30 python ../../dmlc-core/tracker/dmlc-submit"
# build xgboost with librabit mock
# define max worker retry with dmlc-core local num atempt
# instrument worker failure with mock=xxxx
# check if host recovered from expectected iteration
echo "====== 1. Fault recovery distributed test with native ======"
$submit --cluster=local --num-workers=3 --local-num-attempt=20 ../../xgboost machine.conf
