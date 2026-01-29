#!/bin/bash
# Build Python wheels, CPU variant (no federated learning)

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {manylinux2014,manylinux_2_28} {x86_64,aarch64}"
  exit 1
fi

manylinux_target="$1"
arch="$2"

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

WHEEL_TAG="${manylinux_target}_${arch}"
IMAGE_REPO="xgb-ci.${WHEEL_TAG}"
IMAGE_URI="${DOCKER_REGISTRY_URL}/${IMAGE_REPO}:${IMAGE_TAG}"
PYTHON_BIN="/opt/python/cp310-cp310/bin/python"

echo "--- Build binary wheel for ${WHEEL_TAG} (CPU only)"
set -x

# Patch to rename pkg to xgboost-cpu
python3 ops/script/pypi_variants.py --use-suffix=cpu --require-nccl-dep=na
python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- bash -c \
  "cd python-package && ${PYTHON_BIN} -m pip wheel --no-deps -v . --wheel-dir dist/"

python3 ops/docker_run.py \
  --image-uri "${IMAGE_URI}" \
  -- auditwheel repair --only-plat \
  --plat ${WHEEL_TAG} python-package/dist/xgboost_cpu-*.whl
python3 -m wheel tags --python-tag py3 --abi-tag none --platform ${WHEEL_TAG} --remove \
  wheelhouse/xgboost_cpu-*.whl
rm -v python-package/dist/xgboost_cpu-*.whl
mv -v wheelhouse/xgboost_cpu-*.whl python-package/dist/

if ! unzip -l ./python-package/dist/*.whl | grep libgomp > /dev/null; then
  echo "error: libgomp.so was not vendored in the wheel"
  exit -1
fi

# Check size of wheel
pydistcheck --config python-package/pyproject.toml python-package/dist/*.whl

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    python-package/dist/*.whl
fi
