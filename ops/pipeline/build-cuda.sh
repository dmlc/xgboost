#!/bin/bash
## Build XGBoost with CUDA
## This script runs inside a container (via GitHub Actions container support)

set -euo pipefail

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {x86_64,aarch64} {enable-rmm,disable-rmm}"
  exit 2
fi
arch="$1"
rmm_flag="$2"
export USE_FEDERATED=1

# Validate RMM flag
case "${rmm_flag}" in
  enable-rmm)
    export USE_RMM=1
    ;;
  disable-rmm)
    export USE_RMM=0
    ;;
  *)
    echo "Unrecognized argument: $rmm_flag"
    exit 3
    ;;
esac

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
