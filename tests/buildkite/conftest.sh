#!/bin/bash

set -euo pipefail
set -x

CUDA_VERSION=11.0.3
RAPIDS_VERSION=22.10
SPARK_VERSION=3.0.1
JDK_VERSION=8

if [[ -z ${BUILDKITE:-} ]]
then
  echo "$0 is not meant to run locally; it should run inside BuildKite."
  echo "Please inspect the content of $0 and locate the desired command manually."
  exit 1
fi

if [[ -n $BUILDKITE_PULL_REQUEST && $BUILDKITE_PULL_REQUEST != "false" ]]
then
  is_pull_request=1
  export BRANCH_NAME=PR-$BUILDKITE_PULL_REQUEST
else
  is_pull_request=0
  export BRANCH_NAME=$BUILDKITE_BRANCH
fi

if [[ $BUILDKITE_BRANCH == "master" || $BUILDKITE_BRANCH == "release_"* ]]
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
