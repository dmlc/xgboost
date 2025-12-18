#!/bin/bash
## Test XGBoost against latest CCCL

set -euo pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [cccl_version]"
  exit 1
fi
cccl_version="$1"

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_REPO="xgb-ci.gpu_build_cuda13_rockylinux8"
BUILD_IMAGE_URI="${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}"

echo "--- Build XGBoost with CCCL ${cccl_version}"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi
echo "BUILD_ONLY_SM75=${BUILD_ONLY_SM75}"

set -x

python3 ops/docker_run.py \
  --image-uri ${BUILD_IMAGE_URI} \
  --run-args='-e BUILD_ONLY_SM75' \
  -- ops/pipeline/nightly-test-cccl-impl.sh "${cccl_version}"
