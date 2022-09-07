#!/bin/bash

set -euo pipefail

CUDA_VERSION=11.0.3

source tests/buildkite/conftest.sh

echo "--- Build XGBoost R package with CUDA"

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then

tests/ci_build/ci_build.sh gpu_build_r_centos7 docker \
  --build-arg CUDA_VERSION_ARG=${CUDA_VERSION} tests/ci_build/build_r_pkg_with_cuda.sh \
  ${BUILDKITE_COMMIT}

echo "--- Upload R tarball"
aws s3 cp xgboost_r_gpu_linux_*.tar.gz s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
  --acl public-read --no-progress

fi
