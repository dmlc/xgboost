#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

WHEEL_TAG=manylinux2014_aarch64
command_wrapper="tests/ci_build/ci_build.sh manylinux2014_aarch64"
python_bin="/opt/python/cp310-cp310/bin/python"

echo "--- Build binary wheel for ${WHEEL_TAG}"
$command_wrapper bash -c \
  "cd python-package && ${python_bin} -m pip wheel --no-deps -vvv . --wheel-dir dist/"
$command_wrapper auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper ${python_bin} tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
rm -rf python-package/dist/
mkdir python-package/dist/
mv -v wheelhouse/*.whl python-package/dist/

echo "--- Upload Python wheel"
buildkite-agent artifact upload python-package/dist/*.whl
$command_wrapper bash -c "${python_bin} -m pip install awscli"
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  ${python_bin} -m awscli s3 \
    cp python-package/dist/*.whl s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi
