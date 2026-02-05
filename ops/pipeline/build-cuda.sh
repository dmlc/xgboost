#!/bin/bash
## Build XGBoost with CUDA
## This script runs inside a container (via GitHub Actions container support)
##
## Usage:
##   build-cuda.sh --cuda-version <12|13> --use-rmm <0|1> --use-federated <0|1>
##
## All parameters are required (no defaults).
##
## Examples:
##   # CUDA 12 standard build
##   build-cuda.sh --cuda-version 12 --use-rmm 0 --use-federated 1
##
##   # CUDA 12 with RMM
##   build-cuda.sh --cuda-version 12 --use-rmm 1 --use-federated 1
##
##   # CUDA 13
##   build-cuda.sh --cuda-version 13 --use-rmm 0 --use-federated 0
##
##   # Variant wheels (CUDA 12 without federated)
##   build-cuda.sh --cuda-version 12 --use-rmm 0 --use-federated 0

set -euo pipefail

# All parameters are required - no defaults
cuda_version=""
use_rmm=""
use_federated=""

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
    --use-federated)
      use_federated="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 --cuda-version <12|13> --use-rmm <0|1> --use-federated <0|1>"
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

if [[ -z "${use_federated}" ]]; then
  echo "Error: --use-federated is required (0 or 1)"
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

case "${use_federated}" in
  0|1)
    ;;
  *)
    echo "Error: --use-federated must be 0 or 1, got '${use_federated}'"
    exit 1
    ;;
esac

# Validate CUDA 13 constraints
if [[ "${cuda_version}" == "13" ]]; then
  if [[ "${use_rmm}" == "1" ]]; then
    echo "Error: RMM is not supported for CUDA 13 (--use-rmm must be 0)"
    exit 1
  fi
  if [[ "${use_federated}" == "1" ]]; then
    echo "Error: Federated plugin is not supported for CUDA 13 (--use-federated must be 0)"
    exit 1
  fi
fi

# Export validated values
export USE_RMM="${use_rmm}"
export USE_FEDERATED="${use_federated}"

source ops/pipeline/classify-git-branch.sh

echo "--- Build with CUDA ${cuda_version}"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]; then
  export BUILD_ONLY_SM75=1
else
  export BUILD_ONLY_SM75=0
fi

set -x

# Configure PyPI variant for CUDA 13
if [[ "${cuda_version}" == "13" ]]; then
  python3 ops/script/pypi_variants.py --use-suffix=cu13 --require-nccl-dep=cu13
fi

# Run the build implementation directly (we're already inside the container)
ops/pipeline/build-cuda-impl.sh
