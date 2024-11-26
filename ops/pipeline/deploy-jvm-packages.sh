#!/bin/bash
## Deploy JVM packages to S3 bucket

set -euox pipefail

source ops/pipeline/enforce-ci.sh

if [[ "$#" -lt 3 ]]
then
  echo "Usage: $0 {cpu,gpu} [container_id] [scala_version]"
  exit 1
fi

variant="$1"
container_id="$2"
scala_version="$3"

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  python3 ops/docker_run.py --container-id "${container_id}" \
  -- ops/pipeline/deploy-jvm-packages-impl.sh "${variant}" "${scala_version}"
fi
