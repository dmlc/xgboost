#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
pytest -v -s --fulltrace -m "(not mgpu) and (not slow)" tests/python-gpu
pushd .
cd build
./testxgboost --gtest_filter=-*.MGPU_*
ctest --output-on-failure --tests-regex "TestXGBoostCLI"
popd
