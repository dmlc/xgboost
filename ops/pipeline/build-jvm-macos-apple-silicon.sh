#!/bin/bash
## Build libxgboost4j.dylib targeting MacOS (Apple Silicon)

set -euox pipefail

# Display system info
echo "--- Display system information"
set -x
system_profiler SPSoftwareDataType
sysctl -n machdep.cpu.brand_string
uname -m
set +x

brew install ninja libomp

# Build XGBoost4J binary
echo "--- Build libxgboost4j.dylib"
set -x
mkdir build
pushd build
export JAVA_HOME=$(/usr/libexec/java_home)
cmake .. -GNinja -DJVM_BINDINGS=ON -DUSE_OPENMP=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15
ninja -v
popd
rm -rf build
otool -L lib/libxgboost.dylib
set +x
