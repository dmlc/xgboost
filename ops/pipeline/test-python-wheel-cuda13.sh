#!/bin/bash
## Test XGBoost Python wheel on the Linux platform, CUDA 13

set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

ARCH=$(uname -m)
case "${ARCH}" in
  x86_64)
    IMAGE_REPO="xgb-ci.gpu_build_cuda13_rockylinux8"
    ;;
  aarch64)
    IMAGE_REPO="xgb-ci.gpu_build_cuda13_rockylinux8_aarch64"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

IMAGE_URI="${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}"

set -x
python3 ops/docker_run.py --image-uri "${IMAGE_URI}" --use-gpus \
  --run-args='--shm-size=4g --privileged' \
  -- bash ops/pipeline/test-python-wheel-cuda13-impl.sh
