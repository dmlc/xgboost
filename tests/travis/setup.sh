#!/bin/bash

if [ ${TASK} == "python_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    else
	echo "We are no longer running Linux test on Travis."
        exit 1
    fi
    bash conda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    # Useful for debugging any issues with conda
    conda info -a
    conda create -n python3 python=3.7
fi

if [ ${TASK} == "cmake_test" ] && [ ${TRAVIS_OS_NAME} == "osx" ]; then
    sudo softwareupdate -i "Command Line Tools (macOS High Sierra version 10.13) for Xcode-9.3"
fi
