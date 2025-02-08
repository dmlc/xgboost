#!/bin/bash
## Build and test JVM packages.
##
## Note. This script takes in all inputs via environment variables.

INPUT_DOC=$(
cat <<-EOF
Inputs
  - SCALA_VERSION: Scala version, either 2.12 or 2.13 (Required)
EOF
)

set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

for arg in "SCALA_VERSION"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${INPUT_DOC}"
    exit 1
  fi
done

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.jvm:${IMAGE_TAG}

set -x

python3 ops/docker_run.py --image-uri ${IMAGE_URI} \
  --run-args "-e SCALA_VERSION=${SCALA_VERSION}" \
  -- ops/pipeline/build-test-jvm-packages-impl.sh
