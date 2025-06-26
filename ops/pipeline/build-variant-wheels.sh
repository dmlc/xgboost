#!/bin/bash
## Build Python wheels using Wheel Variant prototype (WheelNext)

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 [image_repo]"
  exit 2
fi
image_repo="$1"

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

WHEEL_TAG=manylinux_2_28_x86_64
BUILD_IMAGE_URI="${DOCKER_REGISTRY_URL}/${image_repo}:${IMAGE_TAG}"
MANYLINUX_IMAGE_URI="${DOCKER_REGISTRY_URL}/xgb-ci.${WHEEL_TAG}:${IMAGE_TAG}"

echo "--- Build with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi
export USE_RMM=0

set -x

python3 ops/docker_run.py \
  --image-uri ${BUILD_IMAGE_URI} \
  --run-args='-e BUILD_ONLY_SM75 -e USE_RMM' \
  -- ops/pipeline/build-cuda-impl.sh

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
python3 ops/docker_run.py \
  --image-uri ${MANYLINUX_IMAGE_URI} \
  -- auditwheel repair --only-plat \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
python3 -m wheel tags --python-tag py3 --abi-tag none --platform ${WHEEL_TAG} --remove \
  wheelhouse/*.whl
mv -v wheelhouse/*.whl python-package/dist/

mamba create -n wheelnext python=3.13 python-build
python -m pip install -v git+https://github.com/wheelnext/pep_xxx_wheel_variants.git@f3b287090f8a6f510b0e1723896e1c7e638f6bff#subdirectory=pep_xxx_wheel_variants

pip config set --site global.index-url https://variants-index.wheelnext.dev/
variantlib make-variant -f python-package/dist/xgboost-*.whl -p "nvidia :: cuda :: 12" -o . --pyproject-toml python-package/pyproject.toml
