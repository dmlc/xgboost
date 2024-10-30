#!/bin/bash

set -euo pipefail

source ops/pipeline/enforce-ci.sh

echo "--- Build XGBoost R package with CUDA"
python3 ops/docker_run.py \
  --container-id xgb-ci.gpu_build_r_rockylinux8 \
  -- tests/ci_build/build_r_pkg_with_cuda.sh \
  ${GITHUB_SHA}

if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  echo "--- Upload R tarball"
  aws s3 cp xgboost_r_gpu_linux_*.tar.gz s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi
