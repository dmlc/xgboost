#!/bin/bash

rm -f *.model*

export DMLC_SUBMIT_CLUSTER=local

submit="timeout 30 python ../../dmlc-core/tracker/dmlc-submit"

echo "====== 1. Basic distributed test with Python ======"
$submit --cluster=local --num-workers=3 python test_basic.py

echo "====== 2. Regression test for issue #3402 ======"
$submit --cluster=local --num-workers=2 --worker-cores=1 python test_issue3402.py
