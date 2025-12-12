#!/bin/bash

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [image_repo]"
  exit 2
fi
image_repo="$1"

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/${image_repo}:${IMAGE_TAG}

echo "--- Run Google Tests, using a single GPU, CUDA 13"
python3 ops/docker_run.py --image-uri ${IMAGE_URI} --use-gpus \
  --run-args='--privileged' \
  -- build/testxgboost
