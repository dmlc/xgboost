#!/bin/bash

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

source ops/pipeline/get-docker-registry-details.sh

CONTAINER_TAG=${DOCKER_REGISTRY_URL}/xgb-ci.gpu_build_r_rockylinux8:main

echo "--- Build XGBoost R package with CUDA"
set -x
python3 ops/docker_run.py \
  --container-tag ${CONTAINER_TAG} \
  -- ops/pipeline/build-gpu-rpkg-impl.sh \
  ${GITHUB_SHA}
