#!/bin/bash

## Convenience wrapper for ops/pipeline/stash-artifacts.py
## Meant to be used inside GitHub Actions

set -euo pipefail

source ops/pipeline/enforce-ci.sh

if [[ "$#" -lt 3 ]]
then
  echo "Usage: $0 {stash,unstash} [remote_prefix] [artifact] [artifact ...]"
  exit 1
fi

command="$1"
remote_prefix="$2"
shift 2

for arg in "GITHUB_REPOSITORY" "GITHUB_RUN_ID" "RUNS_ON_S3_BUCKET_CACHE"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo "Error: $arg must be set."
    exit 2
  fi
done

artifact_stash_prefix="cache/${GITHUB_REPOSITORY}/stash/${GITHUB_RUN_ID}"

set -x
python3 ops/pipeline/stash-artifacts.py \
  --command "${command}"  \
  --s3-bucket "${RUNS_ON_S3_BUCKET_CACHE}" \
  --prefix "${artifact_stash_prefix}/${remote_prefix}" \
  -- "$@"
