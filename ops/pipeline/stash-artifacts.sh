#!/bin/bash

## Convenience wrapper for ops/pipeline/stash-artifacts.py
## Meant to be used inside GitHub Actions

ENV_VAR_DOC=$(
cat <<-EOF
Inputs
  - COMMAND: Either "upload" or "download"
  - KEY:     Unique string to identify a group of artifacts
EOF
)

set -euo pipefail

source ops/pipeline/enforce-ci.sh

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [artifact] [artifact ...]"
  exit 1
fi

for arg in "COMMAND" "KEY" "GITHUB_REPOSITORY" "GITHUB_RUN_ID" "RUNS_ON_S3_BUCKET_CACHE"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${ENV_VAR_DOC}"
    exit 1
  fi
done

artifact_stash_prefix="cache/${GITHUB_REPOSITORY}/stash/${GITHUB_RUN_ID}"

set -x
python3 ops/pipeline/stash-artifacts.py \
  --command "${COMMAND}"  \
  --s3-bucket "${RUNS_ON_S3_BUCKET_CACHE}" \
  --prefix "${artifact_stash_prefix}/${KEY}" \
  -- "$@"
