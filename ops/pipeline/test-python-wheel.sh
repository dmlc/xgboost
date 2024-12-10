#!/bin/bash
## Test XGBoost Python wheel on the Linux platform

set -euo pipefail

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} [container_id]"
  exit 1
fi
  
suite="$1"
container_id="$2"

if [[ "$suite" == "gpu" || "$suite" == "mgpu" ]]
then
  gpu_option="--use-gpus"
else
  gpu_option=""
fi

source ops/pipeline/get-docker-registry-details.sh
CONTAINER_TAG="${DOCKER_REGISTRY_URL}/${container_id}:main"

set -x
python3 ops/docker_run.py --container-tag "${CONTAINER_TAG}" ${gpu_option} \
  --run-args='--shm-size=4g --privileged' \
  -- bash ops/pipeline/test-python-wheel-impl.sh "${suite}"
