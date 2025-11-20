#!/bin/bash
## Build XGBoost with CUDA for Linux ARM64

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

IMAGE_REPO="xgb-ci.gpu_build_rockylinux8_aarch64"
export USE_FEDERATED=1
export USE_RMM=0

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

WHEEL_TAG=manylinux_2_28_aarch64
BUILD_IMAGE_URI="${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}"
MANYLINUX_IMAGE_URI="${DOCKER_REGISTRY_URL}/xgb-ci.${WHEEL_TAG}:${IMAGE_TAG}"

echo "--- Build with CUDA (ARM64)"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi

set -x

python3 ops/docker_run.py \
  --image-uri ${BUILD_IMAGE_URI} \
  --run-args='-e BUILD_ONLY_SM75 -e USE_RMM -e USE_FEDERATED' \
  -- ops/pipeline/build-cuda-impl.sh

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --image-uri ${MANYLINUX_IMAGE_URI} \
  -- auditwheel repair --only-plat \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 -m wheel tags --python-tag py3 --abi-tag none --platform ${WHEEL_TAG} --remove \
  wheelhouse/*.whl
mv -v wheelhouse/*.whl python-package/dist/
if ! unzip -l ./python-package/dist/*.whl | grep libgomp > /dev/null; then
  echo "error: libgomp.so was not vendored in the wheel"
  exit -1
fi

# Check size of wheel
pydistcheck --config python-package/pyproject.toml python-package/dist/*.whl

echo "--- Generate meta info"
python3 ops/script/format_wheel_meta.py \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${GITHUB_SHA}  \
  --platform-tag ${WHEEL_TAG}  \
  --meta-path python-package/dist/

echo "--- Upload Python wheel"
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    python-package/dist/*.whl
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME} --make-public \
    python-package/dist/meta.json
fi

