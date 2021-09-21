#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    # https://travis-ci.community/t/macos-build-fails-because-of-homebrew-bundle-unknown-command/7296/27
    # Use libomp 11.1.0: https://github.com/dmlc/xgboost/issues/7039
    brew update  # Force update, so that update doesn't overwrite our version of libomp.rb
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/679923b4eb48a8dc7ecc1f05d06063cd79b3fc00/Formula/libomp.rb -O $(find $(brew --repository) -name libomp.rb)
    brew install cmake libomp
    brew pin libomp
fi



if [ ${TASK} == "python_test" ] || [ ${TASK} == "python_sdist_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        wget --no-verbose -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    elif [ ${TRAVIS_CPU_ARCH} == "arm64" ]; then
        wget --no-verbose -O conda.sh https://github.com/conda-forge/miniforge/releases/download/4.8.2-1/Miniforge3-4.8.2-1-Linux-aarch64.sh
    else
        wget --no-verbose -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    bash conda.sh -b -p $HOME/miniconda
    source $HOME/miniconda/bin/activate
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    # Useful for debugging any issues with conda
    conda info -a
    conda create -n python3 python=3.7 cmake numpy scipy codecov
    conda activate python3
    python -m pip install awscli
fi

if [ ${TASK} == "s390x_test" ] && [ ${TRAVIS_CPU_ARCH} == "s390x" ]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends tar unzip wget git build-essential ninja-build \
	 time python3 python3-pip python3-numpy python3-scipy python3-sklearn r-base
fi
