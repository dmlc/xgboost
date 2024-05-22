#!/bin/bash

set -e
set -x

spark_version=$1
use_cuda=$2
gpu_arch=$3
use_scala213=$4

gpu_options=""
if [ "x$use_cuda" == "x-Duse.cuda=ON" ]; then
  gpu_options="$use_cuda -Pgpu"
fi

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages

if [ "x$gpu_arch" != "x" ]; then
  export GPU_ARCH_FLAG=$gpu_arch
fi

if [ "x$use_scala213" != "x" ]; then
  cd ..
  python dev/change_scala_version.py --scala-version 2.13 --purge-artifacts
  cd jvm-packages
fi

mvn --no-transfer-progress package -Dspark.version=${spark_version} $gpu_options

set +x
set +e
