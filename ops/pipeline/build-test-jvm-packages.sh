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

for arg in "SCALA_VERSION"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${INPUT_DOC}"
    exit 1
  fi
done

set -x

python3 ops/docker_run.py --container-id xgb-ci.jvm \
  --run-args "-e SCALA_VERSION=${SCALA_VERSION}" \
  -- ops/pipeline/build-test-jvm-packages-impl.sh
