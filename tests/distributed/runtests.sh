#!/bin/bash

echo "====== 1. Basic distributed test with Python ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=3\
  python test_basic.py

echo "====== 2. Regression test for issue #3402 ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 --worker-cores=1\
  python test_issue3402.py
