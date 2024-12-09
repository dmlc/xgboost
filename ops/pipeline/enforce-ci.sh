#!/bin/bash

## Ensure that a script is running inside the CI.
## Usage: source ops/pipeline/enforce-ci.sh

set -euo pipefail

if [[ -z ${GITHUB_ACTIONS:-} ]]
then
  echo "$0 is not meant to run locally; it should run inside GitHub Actions."
  echo "Please inspect the content of $0 and locate the desired command manually."
  exit 1
fi

if [[ -z ${BRANCH_NAME:-} ]]
then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 2
fi

source ops/pipeline/classify-git-branch.sh
