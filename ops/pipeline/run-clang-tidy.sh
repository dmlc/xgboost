#!/bin/bash

set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.clang_tidy:${IMAGE_TAG}

echo "--- Run clang-tidy"
set -x
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- python3 ops/script/run_clang_tidy.py --cuda-archs 75
