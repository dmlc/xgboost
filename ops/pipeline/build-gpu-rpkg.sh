#!/bin/bash
## Build XGBoost R package with GPU support and package it in a tarball.
## Users will be able to install it without having CTK installed
## (only a compatible NVIDIA driver is needed).
## This script runs inside the container (via GitHub Actions container support).

set -euox pipefail

python3 ops/script/test_r_package.py --task=pack

gpu_r_lib=$(mktemp -d)
cleanup() {
  rm -r "${gpu_r_lib}"
}
trap cleanup EXIT

env \
  XGBOOST_USE_CUDA=ON \
  CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  R CMD INSTALL --build --clean --library="${gpu_r_lib}" xgboost/

shopt -s nullglob
gpu_packages=(xgboost_*_R_*.tar.gz)
if [[ ${#gpu_packages[@]} -ne 1 ]]; then
  echo "Expected one R binary package, found ${#gpu_packages[@]}: ${gpu_packages[*]}" >&2
  exit 1
fi
mv -v "${gpu_packages[0]}" xgboost_r_gpu_linux.tar.gz
