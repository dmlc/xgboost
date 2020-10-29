#!/bin/bash

set -e
set -x

spark_version=$1
use_cuda=$2
gpu_arch=$3

gpu_options=""
if [ "x$use_cuda" == "x-Duse.cuda=ON" ]; then
  # Since building jvm for CPU will do unit tests, choose gpu-with-gpu-tests profile to build
  gpu_options="$use_cuda -Pgpu-with-gpu-tests "
fi

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages
export RABIT_MOCK=ON

if [ "x$gpu_arch" != "x" ]; then
  export GPU_ARCH_FLAG=$gpu_arch
fi
mvn --no-transfer-progress package -Dspark.version=${spark_version} $gpu_options

set +x
set +e
