#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

WHEEL_TAG=manylinux2014_x86_64
command_wrapper="tests/ci_build/ci_build.sh manylinux2014_x86_64"

echo "--- Build binary wheel for ${WHEEL_TAG}"
$command_wrapper bash -c \
  "cd python-package && pip wheel --no-deps -vvv . --wheel-dir dist/"
$command_wrapper auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper python tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/

echo "--- Upload Python wheel"
buildkite-agent artifact upload python-package/dist/*.whl
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp python-package/dist/*.whl s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi
