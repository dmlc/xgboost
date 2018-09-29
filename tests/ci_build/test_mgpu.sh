#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
python -m nose -v --attr='!slow' tests/python-mgpu/
./testxgboost --gtest_filter=*.MGPU_*
