#!/bin/bash

set -e
set -x

# Bundle libomp 11.1.0 when targeting MacOS.
# This is a workaround in order to prevent segfaults when running inside a Conda environment.
# See https://github.com/dmlc/xgboost/issues/7039#issuecomment-1025125003 for more context.
# The workaround is also used by the scikit-learn project.
if [[ "$RUNNER_OS" == "macOS" ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # XGBoost wheels for MacOS.

    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # arm64 builds must cross compile because CI is on x64
        # cibuildwheel will take care of cross-compilation.
        export PYTHON_CROSSENV=1
        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    else
        export MACOSX_DEPLOYMENT_TARGET=10.13
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
    fi

    sudo conda create -n build $OPENMP_URL
    PREFIX="/usr/local/miniconda/envs/build"

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I$PREFIX/include"
    export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
fi

python -m pip install cibuildwheel
python -m cibuildwheel python-package --output-dir wheelhouse
