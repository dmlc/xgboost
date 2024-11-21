#!/bin/bash
## Detect whether the current git branch is a pull request or a release branch

set -euo pipefail

if [[ -n ${GITHUB_BASE_REF:-} ]]
then
  is_pull_request=1
else
  is_pull_request=0
fi

if [[ ${BRANCH_NAME:-} == "master" || ${BRANCH_NAME:-} == "release_"* || ${BRANCH_NAME:-} == "federated-secure" ]]
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
