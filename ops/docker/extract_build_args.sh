#!/bin/bash
## Extract container definition and build args from ops/docker/ci_container.yml,
## given the container ID.
##
## Example input:
##   xgb-ci.clang_tidy
## Example output:
##   CONTAINER_DEF='clang_tidy' BUILD_ARGS='--build-arg CUDA_VERSION_ARG=12.4.1'

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [container_id]"
    exit 1
fi

CONTAINER_ID="$1"
CONTAINER_DEF=$(
  yq -o json ops/docker/ci_container.yml |
  jq -r --arg container_id "${CONTAINER_ID}" '.[$container_id].container_def'
)
BUILD_ARGS=$(
  yq -o json ops/docker/ci_container.yml |
  jq -r --arg container_id "${CONTAINER_ID}" \
  'include "ops/docker/extract_build_args";
    compute_build_args(.; $container_id)'
)
echo "CONTAINER_DEF='${CONTAINER_DEF}' BUILD_ARGS='${BUILD_ARGS}'"
