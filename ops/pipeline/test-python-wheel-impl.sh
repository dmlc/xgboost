#!/bin/bash
## Companion script for ops/pipeline/test-python-wheel.sh
##
## Usage:
##   test-python-wheel-impl.sh --suite <suite> --cuda-version <12|13>
##
## All parameters are required (no defaults).

set -eo pipefail

suite=""
cuda_version=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      suite="$2"
      shift 2
      ;;
    --cuda-version)
      cuda_version="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 --suite {gpu|mgpu|gpu-arm64|cpu|cpu-arm64} --cuda-version {12|13}"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [[ -z "${suite}" ]]; then
  echo "Error: --suite is required (gpu, mgpu, gpu-arm64, cpu, or cpu-arm64)"
  exit 1
fi

if [[ -z "${cuda_version}" ]]; then
  echo "Error: --cuda-version is required (12 or 13)"
  exit 1
fi

# Validate parameter values
case "${suite}" in
  gpu|mgpu|gpu-arm64|cpu|cpu-arm64)
    ;;
  *)
    echo "Error: --suite must be one of: gpu, mgpu, gpu-arm64, cpu, cpu-arm64. Got '${suite}'"
    exit 1
    ;;
esac

case "${cuda_version}" in
  12|13)
    ;;
  *)
    echo "Error: --cuda-version must be 12 or 13, got '${cuda_version}'"
    exit 1
    ;;
esac

# Set up conda environment based on CUDA version and suite
# Cannot set -u before Conda env activation
if [[ "${cuda_version}" == "13" ]]; then
  # CUDA 13: Create conda environment on-the-fly
  # Fix permissions for conda directories
  gosu root chown -R "$(id -u):$(id -g)" /opt/miniforge/envs /opt/miniforge/pkgs/cache
  gosu root chown "$(id -u):$(id -g)" /opt/miniforge/pkgs
  mamba create -y -n gpu_test python=3.12 pytest cupy scipy numpy pandas scikit-learn joblib hypothesis
  source activate gpu_test
else
  # CUDA 12: Use pre-existing conda environments
  case "$suite" in
    gpu|mgpu|gpu-arm64)
      source activate gpu_test
      ;;
    cpu|cpu-arm64)
      source activate linux_cpu_test
      ;;
    *)
      echo "Unrecognized suite: $suite"
      exit 1
      ;;
  esac
fi

set -xu

export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)
export SPARK_TESTING=1

pip install -v ./wheelhouse/*.whl

case "$suite" in
  gpu)
    echo "-- Run Python tests, using a single GPU"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --durations=0 -m 'not mgpu' tests/python-gpu
    ;;
  gpu-arm64)
    echo "-- Run Python tests, using a single GPU (ARM64)"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --durations=0 -m 'not mgpu' tests/python-gpu
    ;;
  mgpu)
    echo "-- Run Python tests, using multiple GPUs"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --durations=0 -m 'mgpu' tests/python-gpu
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_gpu_with_dask
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_gpu_with_spark
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_gpu_federated
    ;;
  cpu)
    echo "-- Run Python tests (CPU)"
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    pytest -v -s -rxXs --durations=0 tests/python
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_dask
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_spark
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_federated
    ;;
  cpu-arm64)
    echo "-- Run Python tests (CPU, ARM64)"
    pytest -v -s -rxXs --durations=0 \
      tests/python/test_basic.py tests/python/test_basic_models.py \
      tests/python/test_model_compatibility.py
    ;;
  *)
    echo "Unrecognized suite: $suite"
    exit 1
    ;;
esac
