#!/bin/bash

set -euo pipefail

WHEEL_TAG=manylinux2014_aarch64

echo "--- Build CPU code targeting ARM64"

source tests/buildkite/conftest.sh

command_wrapper="tests/ci_build/ci_build.sh aarch64 docker"

echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/build_via_cmake.sh --conda-env=aarch64_test \
  -DUSE_OPENMP=ON -DHIDE_CXX_SYMBOL=ON
echo "--- Run Google Test"
$command_wrapper bash -c "cd build && ctest --extra-verbose"

echo "--- Build binary wheel"
$command_wrapper bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
$command_wrapper python tests/ci_build/rename_whl.py python-package/dist/*.whl \
  ${BUILDKITE_COMMIT} ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with manylinux2014 standard"
$command_wrapper auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper python tests/ci_build/rename_whl.py wheelhouse/*.whl \
  ${BUILDKITE_COMMIT} ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
$command_wrapper bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"

echo "--- Upload Python wheel"
buildkite-agent artifact upload "python-package/dist/*.whl"
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp python-package/dist/*.whl s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi

echo "--- Stash XGBoost CLI executable"
buildkite-agent artifact upload ./xgboost
