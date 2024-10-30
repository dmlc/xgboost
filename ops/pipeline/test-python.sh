#!/bin/bash

set -euo pipefail

source ops/pipeline/enforce-ci.sh

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} {container_id}"
  exit 1
fi
  
suite="$1"
container_id="$2"

tee test-python-wrapper.sh <<-'EOF'
#!/bin/bash
source activate "$1"

set -euox pipefail

export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)
export SPARK_TESTING=1

pip install -v ./python-package/dist/*.whl
EOF

case "$suite" in
  gpu)
    echo "-- Run Python tests, using a single GPU"
    echo "
      python -c 'from cupy.cuda import jitify; jitify._init_module()'
      pytest -v -s -rxXs --fulltrace --durations=0 -m 'not mgpu' tests/python-gpu
    " | tee -a test-python-wrapper.sh
    python3 ops/docker_run.py --container-id "${container_id}" --use-gpus \
      --run-args='--privileged' \
      -- bash test-python-wrapper.sh gpu_test
    ;;

  mgpu)
    echo "-- Run Python tests, using multiple GPUs"
    echo "
      python -c 'from cupy.cuda import jitify; jitify._init_module()'
      pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' tests/python-gpu
      pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' tests/test_distributed/test_gpu_with_dask
      pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' tests/test_distributed/test_gpu_with_spark
      pytest -v -s -rxXs --fulltrace --durations=0 -m 'mgpu' tests/test_distributed/test_gpu_federated
    " | tee -a test-python-wrapper.sh
    python3 ops/docker_run.py --container-id "${container_id}" --use-gpus \
      --run-args='--privileged --shm-size=4g' \
      -- bash test-python-wrapper.sh gpu_test
    ;;

  cpu)
    echo "-- Run Python tests (CPU)"
    echo "
      export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
      pytest -v -s -rxXs --fulltrace --durations=0 tests/python
      pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_dask
      pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_with_spark
      pytest -v -s -rxXs --fulltrace --durations=0 tests/test_distributed/test_federated
    " | tee -a test-python-wrapper.sh
    python3 ops/docker_run.py --container-id "${container_id}" \
      -- bash test-python-wrapper.sh linux_cpu_test
    ;;

  cpu-arm64)
    echo "-- Run Python tests (CPU, ARM64)"
    echo "
      pytest -v -s -rxXs --fulltrace --durations=0 \\
        tests/python/test_basic.py tests/python/test_basic_models.py \\
        tests/python/test_model_compatibility.py
    " | tee -a test-python-wrapper.sh
    python3 ops/docker_run.py --container-id "${container_id}" \
      -- bash test-python-wrapper.sh aarch64_test
    ;;

  *)
    echo "Unrecognized argument: $suite"
    exit 1
    ;;
esac
