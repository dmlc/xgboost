#!/bin/bash
## Build libxgboost4j.dylib for MacOS (Apple Silicon or Intel)

set -euox pipefail

# Display system info
echo "--- Display system information"
system_profiler SPSoftwareDataType
sysctl -n machdep.cpu.brand_string
uname -m

brew install ninja libomp

# Build XGBoost4J binary
echo "--- Build libxgboost4j.dylib"
mkdir build
pushd build
export JAVA_HOME=$(/usr/libexec/java_home)
cmake .. -GNinja \
  -DJVM_BINDINGS=ON \
  -DUSE_OPENMP=ON \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
ninja -v
popd

rm -rf build
otool -L lib/libxgboost.dylib
