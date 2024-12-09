#!/bin/bash

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh

WHEEL_TAG=manylinux_2_28_aarch64
CONTAINER_TAG=${DOCKER_REGISTRY_URL}/xgb-ci.aarch64:main

echo "--- Build CPU code targeting ARM64"
set -x
echo "--- Build libxgboost from the source"
python3 ops/docker_run.py \
  --container-tag ${BUILD_CONTAINER_TAG} \
  -- ops/script/build_via_cmake.sh \
  --conda-env=aarch64_test \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOL=ON

echo "--- Run Google Test"
python3 ops/docker_run.py \
  --container-tag ${BUILD_CONTAINER_TAG} \
  -- bash -c "cd build && ctest --extra-verbose"

echo "--- Build binary wheel"
python3 ops/docker_run.py \
  --container-tag ${BUILD_CONTAINER_TAG} \
  -- bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
python3 ops/script/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --container-tag ${BUILD_CONTAINER_TAG} \
  -- auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/

if ! unzip -l ./python-package/dist/*.whl | grep libgomp > /dev/null; then
  echo "error: libgomp.so was not vendored in the wheel"
  exit -1
fi
