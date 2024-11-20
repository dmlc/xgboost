#!/bin/bash

## Publish artifacts in an S3 bucket
## Meant to be used inside GitHub Actions

set -euo pipefail

source ops/pipeline/enforce-ci.sh

if [[ $# -ne 2 ]]
then
  echo "Usage: $0 [artifact] [s3_url]"
  exit 1
fi

artifact="$1"
s3_url="$2"

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "aws s3 cp ${artifact} ${s3_url} --acl public-read --no-progress"
  aws s3 cp "${artifact}" "${s3_url}" --acl public-read --no-progress
fi
