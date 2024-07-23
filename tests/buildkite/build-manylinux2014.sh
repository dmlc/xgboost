#!/bin/bash

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 {x86_64,aarch64}"
  exit 1
fi

arch=$1

source tests/buildkite/conftest.sh

WHEEL_TAG="manylinux2014_${arch}"
command_wrapper="tests/ci_build/ci_build.sh ${WHEEL_TAG}"
python_bin="/opt/python/cp310-cp310/bin/python"

echo "--- Build binary wheel for ${WHEEL_TAG}"
# Patch to add warning about manylinux2014 variant
patch -p0 < tests/buildkite/remove_nccl_dep.patch
patch -p0 < tests/buildkite/manylinux2014_warning.patch
$command_wrapper bash -c \
  "cd python-package && ${python_bin} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml python-package/xgboost/core.py  # discard the patch

$command_wrapper auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper ${python_bin} tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
rm -rf python-package/dist/
mkdir python-package/dist/
mv -v wheelhouse/*.whl python-package/dist/

echo "--- Build binary wheel for ${WHEEL_TAG} (CPU only)"
# Patch to rename pkg to xgboost-cpu
patch -p0 < tests/buildkite/remove_nccl_dep.patch
patch -p0 < tests/buildkite/cpu_only_pypkg.patch
$command_wrapper bash -c \
  "cd python-package && ${python_bin} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml  # discard the patch

$command_wrapper auditwheel repair --plat ${WHEEL_TAG} python-package/dist/xgboost_cpu-*.whl
$command_wrapper ${python_bin} tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/xgboost_cpu-*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
rm -v python-package/dist/xgboost_cpu-*.whl
mv -v wheelhouse/xgboost_cpu-*.whl python-package/dist/

echo "--- Upload Python wheel"
for wheel in python-package/dist/*.whl
do
  buildkite-agent artifact upload "${wheel}"
done
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  for wheel in python-package/dist/*.whl
  do
    aws s3 cp "${wheel}" s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
      --acl public-read --no-progress
  done
fi
