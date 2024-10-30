#!/bin/bash

set -euox pipefail

echo "--- Run clang-tidy"

source ops/pipeline/enforce-ci.sh

python3 ops/docker_run.py \
  --container-id xgb-ci.clang_tidy \
  -- python3 tests/ci_build/tidy.py --cuda-archs 75
