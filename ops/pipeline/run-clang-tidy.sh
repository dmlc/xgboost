#!/bin/bash

set -euox pipefail

echo "--- Run clang-tidy"

source ops/pipeline/enforce-ci.sh

python3 ops/docker_run.py \
  --container-id xgb-ci.clang_tidy \
  -- python3 ops/run_clang_tidy.py --cuda-archs 75
