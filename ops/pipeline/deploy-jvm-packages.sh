#!/bin/bash
## Deploy JVM packages to S3 bucket

set -euox pipefail

source ops/pipeline/enforce-ci.sh

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {cpu,gpu} {container_id}"
  exit 1
fi

variant="$1"
container_id="$2"

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Deploy JVM packages to xgboost-maven-repo S3 repo"
  python3 ops/docker_run.py --container-id "${container_id}" \
  -- ops/pipeline/deploy-jvm-packages-impl.sh "${variant}"
fi
