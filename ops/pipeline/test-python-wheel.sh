#!/bin/bash
## Test XGBoost Python wheel on the Linux platform

set -euo pipefail

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} [image_repo]"
  exit 1
fi
  
suite="$1"
image_repo="$2"

if [[ "$suite" == "gpu" || "$suite" == "mgpu" ]]
then
  gpu_option="--use-gpus"
else
  gpu_option=""
fi

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI="${DOCKER_REGISTRY_URL}/${image_repo}:${IMAGE_TAG}"

set -x
python3 ops/docker_run.py --image-uri "${IMAGE_URI}" ${gpu_option} \
  --run-args='--shm-size=4g --privileged' \
  -- bash ops/pipeline/test-python-wheel-impl.sh "${suite}"
