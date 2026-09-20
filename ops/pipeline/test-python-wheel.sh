#!/bin/bash
## Script to test Python wheels, to be called from CI workflow
##
## Usage:
##   ops/pipeline/test-python-wheel.sh --suite <suite> [--cuda-version <12|13>]
##
## --suite is required. --cuda-version is required for GPU suites but optional for CPU suites.

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
      echo "Usage: $0 --suite {gpu|mgpu|gpu-arm64|cpu|cpu-arm64} [--cuda-version {12|13}]"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [[ -z "${suite}" ]]; then
  echo "Error: --suite is required (gpu, mgpu, gpu-arm64, cpu, or cpu-arm64)"
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

# Validate --cuda-version is provided for GPU suites
case "${suite}" in
  gpu|mgpu|gpu-arm64)
    if [[ -z "${cuda_version}" ]]; then
      echo "Error: --cuda-version is required for GPU suites (12 or 13)"
      exit 1
    fi
    ;;
esac

# Validate --cuda-version value if provided
if [[ -n "${cuda_version}" ]]; then
  case "${cuda_version}" in
    12|13)
      ;;
    *)
      echo "Error: --cuda-version must be 12 or 13, got '${cuda_version}'"
      exit 1
      ;;
  esac
fi

# Set up the suite's pre-built conda environment
# Cannot set -u before Conda env activation
case "$suite" in
  gpu|mgpu|gpu-arm64)
    source activate gpu_test
    ;;
  cpu|cpu-arm64)
    source activate linux_cpu_test
    ;;
esac

set -xu

export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)
# This variable enables a special test mode in spark, but it's never publicly documented
# as of writing.
export SPARK_TESTING=1

pip install -v ./wheelhouse/*.whl

if [[ -n "${cuda_version}" ]]; then
  python - "${cuda_version}" <<'PY'
import sys

from xgboost import build_info

expected_cuda_major = int(sys.argv[1])
cuda_version = build_info()["CUDA_VERSION"]
if cuda_version[0] != expected_cuda_major:
    raise RuntimeError(
        f"Expected a CUDA {expected_cuda_major} wheel, but it was built with CUDA "
        f"{cuda_version[0]}.{cuda_version[1]}"
    )
print(f"Wheel was built with CUDA {cuda_version[0]}.{cuda_version[1]}")
PY
fi

case "$suite" in
  gpu|gpu-arm64)
    echo "-- Run Python tests, using a single GPU"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --durations=0 -m 'not mgpu' tests/python-gpu
    ;;
  mgpu)
    echo "-- Run Python tests, using multiple GPUs"
    python -c 'from cupy.cuda import jitify; jitify._init_module()'
    pytest -v -s -rxXs --durations=0 -m 'mgpu' tests/python-gpu
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_gpu_with_dask
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_spark/test_data.py -k dmatrix_ctor_gpu
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_spark/test_spark.py -k local_cluster_gpu
    ;;
  cpu)
    echo "-- Run Python tests (CPU)"
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    pytest -v -s -rxXs --durations=0 tests/python
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_dask
    pytest -v -s -rxXs --durations=0 tests/test_distributed/test_with_spark
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
