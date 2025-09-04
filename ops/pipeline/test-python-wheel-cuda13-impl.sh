#!/bin/bash
## Companion script for ops/pipeline/test-python-wheel-cuda13.sh

set -eo pipefail
# Cannot set -u before Conda env activation

# Set up Conda env
mamba create -n gpu_test python=3.12 pytest cupy scipy numpy pandas scikit-learn joblib

source activate gpu_test

set -xu

pip install -v ./wheelhouse/*.whl

echo "-- Run Python tests, using a single GPU, CUDA 13"
python -c 'from cupy.cuda import jitify; jitify._init_module()'
pytest -v -s -rxXs --durations=0 -m 'not mgpu' tests/python-gpu
