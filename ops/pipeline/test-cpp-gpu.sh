#!/bin/bash

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {gpu,gpu-rmm,mgpu}"
  exit 1
fi
arg=$1

case "${arg}" in
  gpu)
    echo "--- Run Google Tests, using a single GPU"
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      -- nvidia-smi
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      -- build/testxgboost
    ;;

  gpu-rmm)
    echo "--- Run Google Tests, using a single GPU, RMM enabled"
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      -- nvidia-smi
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      -- build/testxgboost --use-rmm-pool
    ;;

  mgpu)
    echo "--- Run Google Tests, using multiple GPUs"
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      -- nvidia-smi
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      --run-args='--shm-size=4g' \
      -- build/testxgboost --gtest_filter=*MGPU*
    ;;

  *)
    echo "Unrecognized arg: ${arg}"
    exit 2
    ;;
esac
