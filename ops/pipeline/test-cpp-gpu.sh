#!/bin/bash

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {gpu,gpu-rmm,mgpu}"
  exit 1
fi
suite=$1

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.gpu:${IMAGE_TAG}

case "${suite}" in
  gpu)
    echo "--- Run Google Tests, using a single GPU"
    python3 ops/docker_run.py --image-uri ${IMAGE_URI} --use-gpus \
      --run-args='--privileged' \
      -- build/testxgboost
    ;;

  gpu-rmm)
    echo "--- Run Google Tests, using a single GPU, RMM enabled"
    python3 ops/docker_run.py --image-uri ${IMAGE_URI} --use-gpus \
      --run-args='--privileged' \
      -- build/testxgboost --use-rmm-pool
    ;;

  mgpu)
    echo "--- Run Google Tests, using multiple GPUs"
    python3 ops/docker_run.py --image-uri ${IMAGE_URI} --use-gpus \
      --run-args='--shm-size=4g --privileged' \
      -- build/testxgboost --gtest_filter=*MGPU*
    ;;

  *)
    echo "Unrecognized suite: ${suite}"
    exit 2
    ;;
esac
