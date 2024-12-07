#!/bin/bash
## Run tests on FreeBSD

set -euox pipefail

mkdir build
cd build
cmake .. -GNinja -DGOOGLE_TEST=ON
ninja -v
./testxgboost
