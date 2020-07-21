#!/bin/bash

set -e
set -x

spark_version=$1
use_cuda=$2

gpu_options=""
if [ "x$use_cuda" == "x-Duse.cuda=ON" ]; then
  # Since building jvm for CPU will do unit tests, let's bypass it when building for GPU
  gpu_options=" -Dmaven.test.skip=true -DskipTests $use_cuda "
fi

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages
export RABIT_MOCK=ON
mvn --no-transfer-progress package -Dspark.version=${spark_version} $gpu_options

set +x
set +e
