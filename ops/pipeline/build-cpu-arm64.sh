#!/bin/bash

set -euox pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

WHEEL_TAG=manylinux_2_28_aarch64

echo "--- Build CPU code targeting ARM64"

echo "--- Build libxgboost from the source"
python3 ops/docker_run.py \
  --container-id xgb-ci.aarch64 \
  -- ops/script/build_via_cmake.sh \
  --conda-env=aarch64_test \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOL=ON

echo "--- Run Google Test"
python3 ops/docker_run.py \
  --container-id xgb-ci.aarch64 \
  -- bash -c "cd build && ctest --extra-verbose"

echo "--- Build binary wheel"
python3 ops/docker_run.py \
  --container-id xgb-ci.aarch64 \
  -- bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
python3 ops/script/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --container-id xgb-ci.aarch64 \
  -- auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/

# Make sure that libgomp.so is vendored in the wheel
python3 ops/docker_run.py \
  --container-id xgb-ci.aarch64 \
  -- bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"
