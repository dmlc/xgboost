#!/bin/bash
## Build and test JVM packages.
##
## Note. This script takes in all inputs via environment variables.

INPUT_DOC=$(
cat <<-EOF
Inputs
  - SCALA_VERSION:     Scala version, either 2.12 or 2.13 (Required)
  - USE_CUDA:          Set to 1 to enable CUDA
  - CUDA_ARCH:         Semicolon separated list of GPU compute capability targets
                       (e.g. '35;61') Only applicable if USE_CUDA=1
  - SKIP_NATIVE_BUILD: Set to 1 to have the JVM packages use an externally provided
                       libxgboost4j.so. (Usually Maven will invoke create_jni.py to
                       build it from scratch.) When using this option, make sure to
                       place libxgboost4j.so in lib/ directory.
EOF
)

set -euo pipefail

source ops/pipeline/enforce-ci.sh

for arg in "SCALA_VERSION"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${INPUT_DOC}"
    exit 1
  fi
done

set -x

run_args="-e SCALA_VERSION=${SCALA_VERSION}"
for arg in "USE_CUDA" "CUDA_ARCH" "SKIP_NATIVE_BUILD"
do
  if [[ -n "${!arg:-}" ]]
  then
    run_args="${run_args} -e ${arg}=${!arg}"
  fi
done
echo "${run_args}"

python3 ops/docker_run.py --container-id xgb-ci.jvm \
  --run-args "${run_args}" -- ops/pipeline/build-test-jvm-packages-impl.sh
