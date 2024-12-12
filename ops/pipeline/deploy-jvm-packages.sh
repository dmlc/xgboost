#!/bin/bash
## Deploy JVM packages to S3 bucket

set -euo pipefail

source ops/pipeline/enforce-ci.sh
source ops/pipeline/get-docker-registry-details.sh

if [[ "$#" -lt 3 ]]
then
  echo "Usage: $0 {cpu,gpu} [container_id] [scala_version]"
  exit 1
fi

variant="$1"
container_id="$2"
scala_version="$3"

CONTAINER_TAG="${DOCKER_REGISTRY_URL}/${container_id}:main"

set -x

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  python3 ops/docker_run.py --container-tag "${CONTAINER_TAG}" \
  -- ops/pipeline/deploy-jvm-packages-impl.sh "${variant}" "${scala_version}"
fi
