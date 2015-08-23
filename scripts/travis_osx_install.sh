#!/bin/bash

if [ ${TRAVIS_OS_NAME} != "osx" ]; then
    exit 0
fi

brew update

if [ ${TASK} == "python-package" ]; then
    brew install python git graphviz
    easy_install pip
    pip install numpy scipy matplotlib nose
fi

if [ ${TASK} == "python-package3" ]; then
    brew install python3 git graphviz
    sudo pip3 install --upgrade setuptools
    pip3 install numpy scipy matplotlib nose graphviz
fi
