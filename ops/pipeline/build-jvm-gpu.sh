#!/bin/bash
## Build libxgboost4j.so with CUDA

set -euo pipefail

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh

CONTAINER_TAG=${DOCKER_REGISTRY_URL}/xgb-ci.jvm_gpu_build:main

echo "--- Build libxgboost4j.so with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

COMMAND=$(
cat <<-EOF
cd build-gpu/ && \
cmake .. -DCMAKE_PREFIX_PATH=/workspace/cccl -GNinja -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DJVM_BINDINGS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${arch_flag} && \
  ninja
EOF
)

set -x
mkdir -p build-gpu/
# Work around https://github.com/NVIDIA/cccl/issues/1956
# TODO(hcho3): Remove this once new CUDA version ships with CCCL 2.6.0+
git clone https://github.com/NVIDIA/cccl.git -b v2.6.1 --quiet --depth 1
python3 ops/docker_run.py \
  --container-tag ${CONTAINER_TAG} \
  -- bash -c "${COMMAND}"
