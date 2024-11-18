#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64}"
  exit 1
fi

suite="$1"

set -x

export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)
export SPARK_TESTING=1

pip install -v ./python-package/dist/*.whl

case "$suite" in
  gpu)
    echo "-- Run Python tests, using a single GPU"
    source activate gpu_test
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'not mgpu' tests/python-gpu
    ;;
  mgpu)
    echo "-- Run Python tests, using multiple GPUs"
    source activate gpu_test
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' tests/python-gpu
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' \
      tests/test_distributed/test_gpu_with_dask
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' \
      tests/test_distributed/test_gpu_with_spark
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' \
      tests/test_distributed/test_gpu_federated
    ;;
  cpu)
    echo "-- Run Python tests (CPU)"
    source activate linux_cpu_test
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    pytest -v -s -rxXs --fulltrace --durations=0 tests/python
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_dask
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_spark
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_federated
    ;;
  cpu-arm64)
    echo "-- Run Python tests (CPU, ARM64)"
    source activate aarch64_test
    pytest -v -s -rxXs --fulltrace --durations=0 \
      tests/python/test_basic.py tests/python/test_basic_models.py \
      tests/python/test_model_compatibility.py
    ;;
  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac
