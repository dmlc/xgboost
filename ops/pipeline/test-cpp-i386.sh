#!/bin/bash
## Run C++ tests for i386

set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh

IMAGE_URI="${DOCKER_REGISTRY_URL}/xgb-ci.i386:main"

set -x
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- bash ops/pipeline/test-cpp-i386-impl.sh