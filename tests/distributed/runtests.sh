#!/bin/bash

function check() {
  if [[ $1 -ne 0 ]]; then
    exit $1
  fi
}

echo "====== 1. Basic distributed test with Python ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=3\
  python test_basic.py
diff -q test.model.0 test.model.1 && diff -q test.model.1 test.model.2
check $?

echo "====== 2. Regression test for issue #3402 ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 --worker-cores=1\
  python test_issue3402.py
diff -q test.model.0 test.model.1
check $?
