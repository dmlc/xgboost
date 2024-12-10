#!/bin/bash
## Build libxgboost4j.so targeting glibc 2.17 systems

set -euo pipefail

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 {x86_64,aarch64}"
  exit 1
fi

arch=$1
container_id="xgb-ci.manylinux2014_${arch}"

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh

CONTAINER_TAG="${DOCKER_REGISTRY_URL}/${container_id}:main"

# Build XGBoost4J binary
echo "--- Build libxgboost4j.so (targeting glibc 2.17)"
set -x
mkdir build
python3 ops/docker_run.py \
  --container-tag "${CONTAINER_TAG}" \
  -- bash -c \
  "cd build && cmake .. -DJVM_BINDINGS=ON -DUSE_OPENMP=ON && make -j$(nproc)"
ldd lib/libxgboost4j.so
objdump -T lib/libxgboost4j.so | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -Vu

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  libname=lib/libxgboost4j_linux_${arch}.so
  mv -v lib/libxgboost4j.so ${libname}
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    ${libname}
fi
