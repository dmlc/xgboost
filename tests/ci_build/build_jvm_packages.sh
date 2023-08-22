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
export RABIT_MOCK=ON

if [ "x$gpu_arch" != "x" ]; then
  export GPU_ARCH_FLAG=$gpu_arch
fi

mvn_profile_string=""
if [ "x$use_scala213" != "x" ]; then
  export mvn_profile_string="-Pdefault,scala-2.13"
fi

mvn --no-transfer-progress package $mvn_profile_string -Dspark.version=${spark_version} $gpu_options

set +x
set +e
