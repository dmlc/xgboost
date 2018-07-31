#!/bin/bash

echo "====== 1. Basic distributed test with Python ======"
PYTHONPATH=../../python-package/ ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=3\
  python test_basic.py

echo "====== 2. Regression test for issue #3402 ======"
../../dmlc-core/tracker/dmlc-submit --log-level DEBUG --cluster=local --num-workers=2 --worker-cores=1 ../../xgboost issue3402-example.conf
