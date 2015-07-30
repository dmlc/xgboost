#!/bin/bash

if [ ${TRAVIS_OS_NAME} != "osx" ]; then
    exit 0
fi

brew update

if [ ${TASK} == "python-package" ]; then
    brew install python git
    easy_install pip
    pip install numpy scipy nose
fi
