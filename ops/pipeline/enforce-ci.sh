#!/bin/bash

## Ensure that a script is running inside the CI.
## Usage: source ops/pipeline/enforce-ci.sh

set -euo pipefail

set -x

if [[ -z ${GITHUB_ACTION:-} ]]
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

if [[ -n ${GITHUB_BASE_REF:-} ]]
then
  is_pull_request=1
else
  is_pull_request=0
fi

if [[ $BRANCH_NAME == "master" || $BRANCH_NAME == "release_"* || $BRANCH_NAME == "federated-secure" ]]
then
  is_release_branch=1
  enforce_daily_budget=0
else
  is_release_branch=0
  enforce_daily_budget=1
fi

if [[ -n ${DISABLE_RELEASE:-} ]]
then
  is_release_branch=0
fi

set +x
