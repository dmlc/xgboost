#!/bin/bash

set -e
set -x

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [platform_id] [commit ID]"
  exit 1
fi

platform_id=$1
commit_id=$2

# Bundle libomp 11.1.0 when targeting MacOS.
# This is a workaround in order to prevent segfaults when running inside a Conda environment.
# See https://github.com/dmlc/xgboost/issues/7039#issuecomment-1025125003 for more context.
# The workaround is also used by the scikit-learn project.
if [[ "$platform_id" == macosx_* ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # XGBoost wheels for MacOS.

    if [[ "$platform_id" == macosx_arm64 ]]; then
        # MacOS, Apple Silicon
        # arm64 builds must cross compile because CI is on x64
        # cibuildwheel will take care of cross-compilation.
        wheel_tag=macosx_12_0_arm64
        cpython_ver=38
        setup_env_var='CIBW_TARGET_OSX_ARM64=1'  # extra flag to be passed to xgboost.packager backend
        export PYTHON_CROSSENV=1
        export MACOSX_DEPLOYMENT_TARGET=12.0
        #OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
        OPENMP_URL="https://xgboost-ci-jenkins-artifacts.s3.us-west-2.amazonaws.com/llvm-openmp-11.1.0-hf3c4609_1-osx-arm64.tar.bz2"
    elif [[ "$platform_id" == macosx_x86_64 ]]; then
        # MacOS, Intel
        wheel_tag=macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64
        cpython_ver=38
        export MACOSX_DEPLOYMENT_TARGET=10.13
        #OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
        OPENMP_URL="https://xgboost-ci-jenkins-artifacts.s3.us-west-2.amazonaws.com/llvm-openmp-11.1.0-hda6cdc1_1-osx-64.tar.bz2"
    else
        echo "Platform not supported: $platform_id"
        exit 3
    fi
    # Set up environment variables to configure cibuildwheel
    export CIBW_BUILD=cp${cpython_ver}-${platform_id}
    export CIBW_ARCHS=all
    export CIBW_ENVIRONMENT=${setup_env_var}
    export CIBW_TEST_SKIP='*-macosx_arm64'
    export CIBW_BUILD_VERBOSITY=3

    sudo conda create -n build $OPENMP_URL
    PREFIX="/usr/local/miniconda/envs/build"

    # Set up build flags for cibuildwheel
    # This is needed to bundle libomp lib we downloaded earlier
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I$PREFIX/include"
    export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
else
    echo "Platform not supported: $platform_id"
    exit 2
fi

python -m pip install cibuildwheel
python -m cibuildwheel python-package --output-dir wheelhouse
python tests/ci_build/rename_whl.py wheelhouse/*.whl ${commit_id} ${wheel_tag}
