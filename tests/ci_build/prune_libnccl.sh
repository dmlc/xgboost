#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build

set -x

cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -GNinja
ninja -v PruneLibNccl
mv libnccl_static.a ..
