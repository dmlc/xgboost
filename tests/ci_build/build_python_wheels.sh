#!/bin/bash

set -e
set -x

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [platform_id] [commit ID]"
  exit 1
fi

platform_id=$1
commit_id=$2

if [[ "$platform_id" == macosx_* ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # XGBoost wheels for MacOS.
    if [[ "$platform_id" == macosx_arm64 ]]; then
        # MacOS, Apple Silicon
        # arm64 builds must cross compile because CI is on x64
        # cibuildwheel will take care of cross-compilation.
        wheel_tag=macosx_12_0_arm64
        cpython_ver=39
        cibw_archs=arm64
        export MACOSX_DEPLOYMENT_TARGET=12.0
    elif [[ "$platform_id" == macosx_x86_64 ]]; then
        # MacOS, Intel
        wheel_tag=macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64
        cpython_ver=39
        cibw_archs=x86_64
        export MACOSX_DEPLOYMENT_TARGET=10.15
    else
        echo "Platform not supported: $platform_id"
        exit 3
    fi
    # Set up environment variables to configure cibuildwheel
    export CIBW_BUILD=cp${cpython_ver}-${platform_id}
    export CIBW_ARCHS=${cibw_archs}
    export CIBW_ENVIRONMENT=${setup_env_var}
    export CIBW_TEST_SKIP='*-macosx_arm64'
    export CIBW_BUILD_VERBOSITY=3
else
    echo "Platform not supported: $platform_id"
    exit 2
fi

python -m pip install cibuildwheel
python -m cibuildwheel python-package --output-dir wheelhouse
python tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${commit_id}  \
  --platform-tag ${wheel_tag}
