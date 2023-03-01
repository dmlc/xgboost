#!/bin/bash

set -euo pipefail
set -x

if [ "$#" -lt 1 ]
then
  echo "Usage: $0 [container to build]"
  exit 1
fi
container=$1

source tests/buildkite/conftest.sh

echo "--- Build container ${container}"

BUILD_ARGS=""

case "${container}" in
  cpu)
    ;;

  gpu|rmm)
    BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_VERSION_ARG=$CUDA_VERSION"
    BUILD_ARGS="$BUILD_ARGS --build-arg RAPIDS_VERSION_ARG=$RAPIDS_VERSION"
    if [[ $container == "rmm" ]]
    then
      BUILD_ARGS="$BUILD_ARGS --build-arg NCCL_VERSION_ARG=$NCCL_VERSION"
    fi
    ;;

  gpu_build_centos7|jvm_gpu_build)
    BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_VERSION_ARG=$CUDA_VERSION"
    BUILD_ARGS="$BUILD_ARGS --build-arg NCCL_VERSION_ARG=$NCCL_VERSION"
    ;;

  *)
    echo "Unrecognized container ID: ${container}"
    exit 2
    ;;
esac

# Run a no-op command. This will simply build the container and push it to the private registry
tests/ci_build/ci_build.sh ${container} docker ${BUILD_ARGS} bash
