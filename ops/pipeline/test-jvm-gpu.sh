#!/bin/bash
## Test JVM packages with CUDA. Note: this script assumes that
## the user has already built libxgboost4j.so with CUDA support
## and place it in the lib/ directory.

## Note. This script takes in all inputs via environment variables.

INPUT_DOC=$(
cat <<-EOF
Inputs
  - SCALA_VERSION: Scala version, either 2.12 or 2.13 (Required)
EOF
)

set -euo pipefail

for arg in "SCALA_VERSION"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${INPUT_DOC}"
    exit 1
  fi
done

source ops/pipeline/get-docker-registry-details.sh

CONTAINER_TAG=${DOCKER_REGISTRY_URL}/xgb-ci.jvm_gpu_build:main

set -x

python3 ops/docker_run.py --container-tag ${CONTAINER_TAG} --use-gpus \
  --run-args "-e SCALA_VERSION=${SCALA_VERSION} -e USE_CUDA=1 -e SKIP_NATIVE_BUILD=1 --shm-size=4g --privileged" \
  -- ops/pipeline/build-test-jvm-packages-impl.sh
