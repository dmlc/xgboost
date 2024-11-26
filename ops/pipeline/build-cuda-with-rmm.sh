#!/bin/bash
## Build XGBoost with CUDA + RMM support

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [container_id]"
  exit 1
fi
container_id="$1"

source ops/pipeline/classify-git-branch.sh

set -x

WHEEL_TAG=manylinux_2_28_x86_64

echo "--- Build with CUDA with RMM"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

echo "--- Build libxgboost from the source"
python3 ops/docker_run.py \
  --container-id "${container_id}" \
  -- ops/script/build_via_cmake.sh \
  -DCMAKE_PREFIX_PATH="/opt/grpc;/opt/rmm;/opt/rmm/lib64/rapids/cmake" \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOLS=ON \
  -DPLUGIN_FEDERATED=ON \
  -DPLUGIN_RMM=ON \
  -DUSE_NCCL=ON \
  -DUSE_NCCL_LIB_PATH=ON \
  -DNCCL_INCLUDE_DIR=/usr/include \
  -DUSE_DLOPEN_NCCL=ON \
  ${arch_flag}

echo "--- Build binary wheel"
python3 ops/docker_run.py \
  --container-id "${container_id}" \
  -- bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
python3 ops/script/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --container-id xgb-ci.${WHEEL_TAG} \
  -- auditwheel repair \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
python3 ops/docker_run.py \
  --container-id xgb-ci.${WHEEL_TAG} \
  -- bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"
