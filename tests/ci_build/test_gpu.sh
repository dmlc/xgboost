#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
pytest -m -s --fulltrace "(not mgpu) and (not slow)" tests/python-gpu
./testxgboost --gtest_filter=-*.MGPU_*
