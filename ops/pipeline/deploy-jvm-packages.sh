#!/bin/bash
## Deploy JVM packages to S3 bucket

set -euo pipefail

source ops/pipeline/enforce-ci.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

if [[ "$#" -lt 3 ]]
then
  echo "Usage: $0 {cpu,gpu} [image_repo] [scala_version]"
  exit 1
fi

variant="$1"
image_repo="$2"
scala_version="$3"

IMAGE_URI="${DOCKER_REGISTRY_URL}/${image_repo}:${IMAGE_TAG}"

set -x

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  python3 ops/docker_run.py --image-uri "${IMAGE_URI}" \
  -- ops/pipeline/deploy-jvm-packages-impl.sh "${variant}" "${scala_version}"
fi
