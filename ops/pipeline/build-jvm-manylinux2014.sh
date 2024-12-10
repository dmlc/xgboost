#!/bin/bash
## Build libxgboost4j.so targeting glibc 2.17 systems

set -euox pipefail

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 {x86_64,aarch64}"
  exit 1
fi

arch=$1

image="xgb-ci.manylinux2014_${arch}"

# Build XGBoost4J binary
echo "--- Build libxgboost4j.so (targeting glibc 2.17)"
set -x
mkdir build
python3 ops/docker_run.py \
  --container-id ${image} \
  -- bash -c \
  "cd build && cmake .. -DJVM_BINDINGS=ON -DUSE_OPENMP=ON && make -j$(nproc)"
ldd lib/libxgboost4j.so
objdump -T lib/libxgboost4j.so | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -Vu
