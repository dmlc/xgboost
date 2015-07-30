#!/bin/bash

if [ ${TRAVIS_OS_NAME} != "osx" ]; then
    exit 0
fi

brew update
brew install unzip
brew install python-numpy
brew install python-scipy
brew install python-nose
