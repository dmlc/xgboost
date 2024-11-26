#!/bin/bash
## Build XGBoost with CUDA

set -euox pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

WHEEL_TAG=manylinux_2_28_x86_64

source ops/pipeline/classify-git-branch.sh

echo "--- Build with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

echo "--- Build libxgboost from the source"
set -x
# Work around https://github.com/NVIDIA/cccl/issues/1956
# TODO(hcho3): Remove this once new CUDA version ships with CCCL 2.6.0+
git clone https://github.com/NVIDIA/cccl.git -b v2.6.1 --quiet
python3 ops/docker_run.py \
  --container-id xgb-ci.gpu_build_rockylinux8 \
  -- ops/script/build_via_cmake.sh \
  -DCMAKE_PREFIX_PATH="/opt/grpc;/workspace/cccl" \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DHIDE_CXX_SYMBOLS=ON \
  -DPLUGIN_FEDERATED=ON \
  -DUSE_NCCL=ON \
  -DUSE_NCCL_LIB_PATH=ON \
  -DNCCL_INCLUDE_DIR=/usr/include \
  -DUSE_DLOPEN_NCCL=ON \
  ${arch_flag}

echo "--- Build binary wheel"
python3 ops/docker_run.py \
  --container-id xgb-ci.gpu_build_rockylinux8 \
  -- bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
python3 ops/script/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --container-id xgb-ci.manylinux_2_28_x86_64 \
  -- auditwheel repair \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 ops/script/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
python3 ops/docker_run.py \
  --container-id xgb-ci.manylinux_2_28_x86_64 \
  -- bash -c "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"

# Generate the meta info which includes xgboost version and the commit info
python3 ops/docker_run.py \
--container-id xgb-ci.gpu_build_rockylinux8 \
-- python ops/script/format_wheel_meta.py \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}  \
  --meta-path python-package/dist/

echo "--- Upload Python wheel"
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp python-package/dist/*.whl s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
  aws s3 cp python-package/dist/meta.json s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi
