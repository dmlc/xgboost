#!/bin/bash
## Build libxgboost4j.so with CUDA

set -euo pipefail

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.jvm_gpu_build:${IMAGE_TAG}

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
cmake .. -GNinja -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DJVM_BINDINGS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${arch_flag} && \
  ninja
EOF
)

set -x
mkdir -p build-gpu/
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- bash -c "${COMMAND}"
