#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Build JVM packages doc"
tests/ci_build/ci_build.sh jvm docker tests/ci_build/build_jvm_doc.sh ${BRANCH_NAME}
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Upload JVM packages doc"
  aws s3 cp jvm-packages/${BRANCH_NAME}.tar.bz2 \
    s3://xgboost-docs/${BRANCH_NAME}.tar.bz2 --acl public-read --no-progress
fi
