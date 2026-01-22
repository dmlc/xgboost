#!/bin/bash
## Build Python wheels using Wheel Variant prototype (WheelNext)
## This script runs inside a container (via GitHub Actions container support)

set -euo pipefail

export USE_RMM=0
export USE_FEDERATED=0

source ops/pipeline/classify-git-branch.sh

echo "--- Build with CUDA"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi

set -x

# Run the build implementation directly (we're already inside the container)
ops/pipeline/build-cuda-impl.sh
