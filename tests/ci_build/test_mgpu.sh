#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
pytest -s --fulltrace -m "not slow" tests/python-gpu
./testxgboost --gtest_filter=*.MGPU_*
