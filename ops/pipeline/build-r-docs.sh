#!/bin/bash

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]
then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

source ops/pipeline/get-docker-registry-details.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.cpu_build_r_doc:main

echo "--- Build R package doc"
set -x
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- ops/pipeline/build-r-docs-impl.sh ${BRANCH_NAME}
