#!/usr/bin/env bash
set -e

cd python-package
python setup.py install --user
cd ..
pytest -v -s --fulltrace -m "(not slow) and mgpu" tests/python-gpu
./testxgboost --gtest_filter=*.MGPU_*

cd ../tests/distributed-gpu
./runtests-gpu.sh