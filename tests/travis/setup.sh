#!/bin/bash

if [ ${TASK} == "python_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    else
        wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
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

if [ ${TASK} == "xgb_recovery" ]; then
  if [[ ${TRAVIS_OS_NAME} == "linux" ]]; then sudo apt-get install python3-pip; fi
  if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then
    brew install python3;
    sudo softwareupdate -i "Command Line Tools (macOS High Sierra version 10.13) for Xcode-9.3"
  fi
  sudo pip3 install cpplint pylint urllib3 numpy
  sudo pip3 install websocket-client kubernetes
fi
