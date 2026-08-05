#!/bin/bash
## Build XGBoost with CUDA
## This script runs inside a container (via GitHub Actions container support)
##
## Usage:
##   build-cuda.sh --cuda-version <12|13> --use-rmm <0|1>
##
## All parameters are required (no defaults).
##
## Examples:
##   # CUDA 12 xgboost-cu12 package build
##   build-cuda.sh --cuda-version 12 --use-rmm 0
##
##   # Default CUDA 13 build
##   build-cuda.sh --cuda-version 13 --use-rmm 0
##
##   # CUDA 13 with RMM
##   build-cuda.sh --cuda-version 13 --use-rmm 1

set -euo pipefail

# All parameters are required - no defaults
cuda_version=""
use_rmm=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version)
      cuda_version="$2"
      shift 2
      ;;
    --use-rmm)
      use_rmm="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 --cuda-version <12|13> --use-rmm <0|1>"
      exit 1
      ;;
  esac
done

# Validate all required parameters are provided
if [[ -z "${cuda_version}" ]]; then
  echo "Error: --cuda-version is required (12 or 13)"
  exit 1
fi

if [[ -z "${use_rmm}" ]]; then
  echo "Error: --use-rmm is required (0 or 1)"
  exit 1
fi

# Validate parameter values
case "${cuda_version}" in
  12|13)
    ;;
  *)
    echo "Error: --cuda-version must be 12 or 13, got '${cuda_version}'"
    exit 1
    ;;
esac

case "${use_rmm}" in
  0|1)
    ;;
  *)
    echo "Error: --use-rmm must be 0 or 1, got '${use_rmm}'"
    exit 1
    ;;
esac

if [[ "${cuda_version}" == "12" && "${use_rmm}" == "1" ]]; then
  echo "Error: CUDA 12 is reserved for the xgboost-cu12 package build and does not support RMM"
  exit 1
fi

# Export validated values
export USE_RMM="${use_rmm}"

source ops/pipeline/classify-git-branch.sh

echo "--- Build with CUDA ${cuda_version}"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]; then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi

set -x

# Configure the PyPI package variant
if [[ "${cuda_version}" == "12" ]]; then
  python3 ops/script/pypi_variants.py --use-suffix=cu12 --require-nccl-dep=cu12
else
  python3 ops/script/pypi_variants.py --use-suffix=na --require-nccl-dep=cu13
fi

# Run the build implementation directly (we're already inside the container)
ops/pipeline/build-cuda-impl.sh
