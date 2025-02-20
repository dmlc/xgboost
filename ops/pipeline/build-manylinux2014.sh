#!/bin/bash

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {x86_64,aarch64}"
  exit 1
fi

arch="$1"

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

WHEEL_TAG="manylinux2014_${arch}"
IMAGE_REPO="xgb-ci.${WHEEL_TAG}"
IMAGE_URI="${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}"
PYTHON_BIN="/opt/python/cp310-cp310/bin/python"

echo "--- Build binary wheel for ${WHEEL_TAG}"
set -x
# Patch to add warning about manylinux2014 variant
patch -p0 < ops/patch/remove_nccl_dep.patch
patch -p0 < ops/patch/manylinux2014_warning.patch
python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- bash -c \
  "cd python-package && ${PYTHON_BIN} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml python-package/xgboost/core.py
  # discard the patch

python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- auditwheel repair --only-plat \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 -m wheel tags --python-tag py3 --abi-tag none --platform ${WHEEL_TAG} --remove \
  wheelhouse/*.whl
rm -rf python-package/dist/
mkdir python-package/dist/
mv -v wheelhouse/*.whl python-package/dist/

echo "--- Build binary wheel for ${WHEEL_TAG} (CPU only)"
# Patch to rename pkg to xgboost-cpu
patch -p0 < ops/patch/remove_nccl_dep.patch
patch -p0 < ops/patch/cpu_only_pypkg.patch
python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- bash -c \
  "cd python-package && ${PYTHON_BIN} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml  # discard the patch

python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- auditwheel repair --only-plat \
  --plat ${WHEEL_TAG} python-package/dist/xgboost_cpu-*.whl
python3 -m wheel tags --python-tag py3 --abi-tag none --platform ${WHEEL_TAG} --remove \
  wheelhouse/xgboost_cpu-*.whl
rm -v python-package/dist/xgboost_cpu-*.whl
mv -v wheelhouse/xgboost_cpu-*.whl python-package/dist/

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    python-package/dist/*.whl
fi
