#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
python -m nose -v --eval-attr='(not slow) and mgpu' tests/python-gpu/
./testxgboost --gtest_filter=*.MGPU_*
