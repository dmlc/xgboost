#!/bin/bash

set -euo pipefail

source ops/task/enforce-ci.sh

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {build-cuda,build-cuda-with-rmm}"
  exit 1
fi
arg=$1

case "${arg}" in
  build-cuda)
    echo "--- Run Google Tests with CUDA, using a GPU"
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      --run-args='--privileged' \
      -- build/testxgboost
    ;;

  build-cuda-with-rmm)
    echo "--- Run Google Tests with CUDA, using a GPU, RMM enabled"
    python3 ops/docker_run.py --container-id xgb-ci.gpu --use-gpus \
      --run-args='--privileged' \
      -- build/testxgboost --use-rmm-pool
    ;;

  *)
    echo "Unrecognized arg: ${arg}"
    exit 2
    ;;
esac
