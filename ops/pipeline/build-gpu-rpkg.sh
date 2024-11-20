#!/bin/bash

set -euox pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

echo "--- Build XGBoost R package with CUDA"
python3 ops/docker_run.py \
  --container-id xgb-ci.gpu_build_r_rockylinux8 \
  -- ops/pipeline/build-gpu-rpkg-impl.sh \
  ${GITHUB_SHA}
