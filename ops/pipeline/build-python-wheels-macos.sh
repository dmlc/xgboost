#!/bin/bash
# Build Python wheels targeting MacOS (no federated learning)

set -euox pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [platform_id] [commit ID]"
  exit 1
fi

platform_id=$1
commit_id=$2

if [[ "$platform_id" == macosx_* ]]; then
    if [[ "$platform_id" == macosx_arm64 ]]; then
        # MacOS, Apple Silicon
        cpython_ver=310
        cibw_archs=arm64
        export MACOSX_DEPLOYMENT_TARGET=12.0
    elif [[ "$platform_id" == macosx_x86_64 ]]; then
        # MacOS, Intel
        cpython_ver=310
        cibw_archs=x86_64
        export MACOSX_DEPLOYMENT_TARGET=10.15
        export CIBW_CONFIG_SETTINGS='use_openmp=false'
    else
        echo "Platform not supported: $platform_id"
        exit 3
    fi
    # Set up environment variables to configure cibuildwheel
    export CIBW_BUILD=cp${cpython_ver}-${platform_id}
    export CIBW_ARCHS=${cibw_archs}
    export CIBW_TEST_SKIP='*-macosx_arm64'
    export CIBW_BUILD_VERBOSITY=3
else
    echo "Platform not supported: $platform_id"
    exit 2
fi

# Tell delocate-wheel to not vendor libomp.dylib into the wheel
export CIBW_REPAIR_WHEEL_COMMAND_MACOS="delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel} --exclude libomp.dylib"

env | grep -E '^(CC|CXX|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|SDKROOT|DEVELOPER_DIR|MACOSX_DEPLOYMENT_TARGET|PATH|CONDA_PREFIX)=' || true
which clang || true
which c++ || true
/usr/bin/clang --version || true
/usr/bin/c++ --version || true
xcode-select -p || true
brew list --versions libomp llvm llvm@18 || true
ls -l /usr/local/opt/libomp /usr/local/opt/llvm@18 /usr/local/Cellar/llvm@18 || true
brew unlink llvm@18 || true
brew list --versions libomp llvm llvm@18 || true
ls -l /usr/local/opt/libomp /usr/local/opt/llvm@18 /usr/local/Cellar/llvm@18 || true

python -m pip install cibuildwheel
python -m cibuildwheel python-package --output-dir wheelhouse
