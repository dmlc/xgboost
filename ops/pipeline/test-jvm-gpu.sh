#!/bin/bash
## Test JVM packages with CUDA. Note: this script assumes that
## the user has already built libxgboost4j.so with CUDA support
## and place it in the lib/ directory.

set -euo pipefail

# source ops/pipeline/enforce-ci.sh

SCALA_VERSION=2.12

set -x

python3 ops/docker_run.py --container-id xgb-ci.jvm_gpu_build --use-gpus \
  -- nvidia-smi
python3 ops/docker_run.py --container-id xgb-ci.jvm_gpu_build --use-gpus \
  --run-args "-e SCALA_VERSION=${SCALA_VERSION} -e USE_CUDA=1 -e SKIP_NATIVE_BUILD=1" \
  -- ops/pipeline/build-test-jvm-packages-impl.sh
