#!/bin/bash
## Companion script for ops/pipeline/test-python-wheel.sh

set -eo pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64}"
  exit 1
fi

suite="$1"

# Cannot set -u before Conda env activation
case "$suite" in
  gpu|mgpu)
    source activate gpu_test
    ;;
  cpu)
    source activate linux_cpu_test
    ;;
  cpu-arm64)
    source activate aarch64_test
    ;;
  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac

set -xu

export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)
export SPARK_TESTING=1

pip install -v ./wheelhouse/*.whl

case "$suite" in
  gpu)
    echo "-- Run Python tests, using a single GPU"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --fulltrace --durations=0 -m 'not mgpu' tests/python-gpu
    ;;
  mgpu)
    echo "-- Run Python tests, using multiple GPUs"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    export NCCL_RAS_ENABLE=0
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
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    pytest -v -s -rxXs --fulltrace --durations=0 tests/python
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_dask
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_spark
    pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_federated
    ;;
  cpu-arm64)
    echo "-- Run Python tests (CPU, ARM64)"
    pytest -v -s -rxXs --fulltrace --durations=0 \
      tests/python/test_basic.py tests/python/test_basic_models.py \
      tests/python/test_model_compatibility.py
    ;;
  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac
