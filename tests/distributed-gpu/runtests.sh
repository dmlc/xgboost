#!/bin/bash

echo "====== 1. Basic GPU distributed test with Python ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2\
  python test_basic.py

