#!/bin/bash

set -euo pipefail

function get_aws_secret {
  if [[ $# -ne 1 ]]
  then
    echo "Usage: get_aws_secret [Name of secret]"
    return 1
  fi
  aws secretsmanager get-secret-value --secret-id $1 --output text --region us-west-2 --query SecretString
}

function set_buildkite_env_vars_in_container {
  # Pass all Buildkite-specific env vars to Docker containers.
  # This is to be used with tests/ci_build/ci_build.sh
  export CI_DOCKER_EXTRA_PARAMS_INIT="${CI_DOCKER_EXTRA_PARAMS_INIT:-} "`
    `"--env BUILDKITE_ANALYTICS_TOKEN --env BUILDKITE_BUILD_ID --env BUILDKITE_BUILD_NUMBER "`
    `"--env BUILDKITE_JOB_ID --env BUILDKITE_BRANCH --env BUILDKITE_COMMIT "`
    `"--env BUILDKITE_MESSAGE --env BUILDKITE_BUILD_URL"
}

set -x

CUDA_VERSION=12.8.0
NCCL_VERSION=2.25.1-1
RAPIDS_VERSION=24.12
DEV_RAPIDS_VERSION=24.06
SPARK_VERSION=3.5.1
JDK_VERSION=8
R_VERSION=4.3.2

if [[ -z ${BUILDKITE:-} ]]
then
  echo "$0 is not meant to run locally; it should run inside BuildKite."
  echo "Please inspect the content of $0 and locate the desired command manually."
  exit 1
fi

if [[ -n $BUILDKITE_PULL_REQUEST && $BUILDKITE_PULL_REQUEST != "false" ]]
then
  is_pull_request=1
  BRANCH_NAME=PR-$BUILDKITE_PULL_REQUEST
else
  is_pull_request=0
  BRANCH_NAME=$BUILDKITE_BRANCH
fi
export BRANCH_NAME=${BRANCH_NAME//\//-}

if [[ $BRANCH_NAME == "master" || $BRANCH_NAME == "release_"* ]]
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
