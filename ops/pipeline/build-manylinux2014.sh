#!/bin/bash

set -euox pipefail

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

WHEEL_TAG="manylinux2014_${arch}"
image="xgb-ci.${WHEEL_TAG}"

python_bin="/opt/python/cp310-cp310/bin/python"

echo "--- Build binary wheel for ${WHEEL_TAG}"
# Patch to add warning about manylinux2014 variant
patch -p0 < ops/patch/remove_nccl_dep.patch
patch -p0 < ops/patch/manylinux2014_warning.patch
python3 ops/docker_run.py \
  --container-id ${image} \
  -- bash -c \
  "cd python-package && ${python_bin} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml python-package/xgboost/core.py
  # discard the patch

python3 ops/docker_run.py \
  --container-id ${image} \
  -- auditwheel repair --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
rm -rf python-package/dist/
mkdir python-package/dist/
mv -v wheelhouse/*.whl python-package/dist/

echo "--- Build binary wheel for ${WHEEL_TAG} (CPU only)"
# Patch to rename pkg to xgboost-cpu
patch -p0 < ops/patch/remove_nccl_dep.patch
patch -p0 < ops/patch/cpu_only_pypkg.patch
python3 ops/docker_run.py \
  --container-id ${image} \
  -- bash -c \
  "cd python-package && ${python_bin} -m pip wheel --no-deps -v . --wheel-dir dist/"
git checkout python-package/pyproject.toml  # discard the patch

python3 ops/docker_run.py \
  --container-id ${image} \
  -- auditwheel repair --plat ${WHEEL_TAG} python-package/dist/xgboost_cpu-*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/xgboost_cpu-*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
rm -v python-package/dist/xgboost_cpu-*.whl
mv -v wheelhouse/xgboost_cpu-*.whl python-package/dist/
