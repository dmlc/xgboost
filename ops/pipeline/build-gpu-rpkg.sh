#!/bin/bash

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.gpu_build_r_rockylinux8:${IMAGE_TAG}

echo "--- Build XGBoost R package with CUDA"
set -x
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- ops/pipeline/build-gpu-rpkg-impl.sh \
  ${GITHUB_SHA}

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    xgboost_r_gpu_linux.tar.gz
fi
