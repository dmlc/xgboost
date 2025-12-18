#!/bin/bash
## Test XGBoost Python wheel on the Linux platform, CUDA 13

set -euo pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [image_repo]"
  exit 2
fi
image_repo="$1"

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI="${DOCKER_REGISTRY_URL}/${image_repo}:${IMAGE_TAG}"

set -x
python3 ops/docker_run.py --image-uri "${IMAGE_URI}" --use-gpus \
  --run-args='--shm-size=4g --privileged' \
  -- bash ops/pipeline/test-python-wheel-cuda13-impl.sh
