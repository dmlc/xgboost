#!/bin/bash

## Stash an artifact in an S3 bucket for later use
##
## Note. This script takes in all inputs via environment variables
##       except the path to the artifact(s).

set -euo pipefail

ENV_VAR_DOC=$(
cat <<-EOF
Inputs
  - COMMAND:   Whether to upload or download the artifact. Either "upload" or
               "download"
  - S3_BUCKET: Name of the S3 bucket to store the artifact
  - PREFIX:    Where the artifact would be stored. The artifact will be stored
               in s3://{S3_BUCKET}/{prefix}/.
EOF
)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [artifact] [artifact ...]"
    exit 1
fi

for arg in "COMMAND" "S3_BUCKET" "PREFIX"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${ENV_VAR_DOC}"
    exit 1
  fi
done

compute_s3_url() {  # args: artifact
  S3_URL="s3://${S3_BUCKET}/${PREFIX}/"$(basename "$1")
}

aws_s3_cp() {  # args: src, dest
  set -x
  aws s3 cp --no-progress "$1" "$2"
  set +x
  return 0
}

if [[ "$COMMAND" == "upload" ]]
then
  echo "Uploading artifacts with prefix $PREFIX..."
  for artifact in "$@"
  do
    compute_s3_url "${artifact}"
    aws_s3_cp "${artifact}" "${S3_URL}"
  done
elif [[ "$COMMAND" == "download" ]]
then
  echo "Downloading artifacts with prefix $PREFIX..."
  for artifact in "$@"
  do
    compute_s3_url "${artifact}"
    aws_s3_cp "${S3_URL}" "${artifact}"
  done
else
  echo "Unrecognized command: $COMMAND"
  exit 2
fi

