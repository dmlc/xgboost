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

brew list --versions libomp llvm llvm@18 || true
ls -l /usr/local/opt/libomp /usr/local/opt/llvm@18 /usr/local/Cellar/llvm@18 || true

python -m pip install cibuildwheel
python -m cibuildwheel python-package --output-dir wheelhouse

# List dependencies of libxgboost.dylib
mkdir tmp
unzip -j wheelhouse/xgboost-*.whl xgboost/lib/libxgboost.dylib -d tmp
otool -L tmp/libxgboost.dylib
otool -l tmp/libxgboost.dylib | grep -E 'cmd LC_(LOAD|REEXPORT|BUILD_VERSION|VERSION_MIN_MACOSX)' -A3 || true
rm -rf tmp
