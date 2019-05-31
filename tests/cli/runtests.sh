#!/bin/bash

rm -f *.model*

export DMLC_SUBMIT_CLUSTER=local

submit="timeout 30 python ../../dmlc-core/tracker/dmlc-submit"

echo "====== 1. Basic distributed test with Python ======"
$submit --cluster=local --num-workers=3 ../../xgboost machine.conf
