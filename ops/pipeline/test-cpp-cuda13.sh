#!/bin/bash

set -euox pipefail

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_REPO='xgb-ci.gpu_build_cuda13_rockylinux8'
IMAGE_URI=${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}

echo "--- Run Google Tests, using a single GPU, CUDA 13"
python3 ops/docker_run.py --image-uri ${IMAGE_URI} --use-gpus \
  --run-args='--privileged' \
  -- build/testxgboost
