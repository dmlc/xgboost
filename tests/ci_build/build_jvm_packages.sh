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

# Purge artifacts and set correct Scala version
pushd ..
if [ "x$use_scala213" != "x" ]; then
  python dev/change_scala_version.py --scala-version 2.13 --purge-artifacts
else
  python dev/change_scala_version.py --scala-version 2.12 --purge-artifacts
fi
popd

mvn --no-transfer-progress package -Dspark.version=${spark_version} $gpu_options

set +x
set +e
