#!/bin/bash
## Build XGBoost with CUDA 13
## This script runs inside a container (via GitHub Actions container support)

set -euo pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {x86_64,aarch64}"
  exit 2
fi
arch="$1"
export USE_RMM=0
export USE_FEDERATED=0

source ops/pipeline/classify-git-branch.sh

echo "--- Build with CUDA 13"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi

set -x

# Configure for CUDA 13 variant
python3 ops/script/pypi_variants.py --use-suffix=cu13 --require-nccl-dep=cu13

# Run the build implementation directly (we're already inside the container)
ops/pipeline/build-cuda-impl.sh
