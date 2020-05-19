#!/bin/bash
set -e
set -x

if [ ${TRAVIS_OS_NAME} != "osx" ]; then
    exit 0
fi

# Prevent clash between Python 2 and 3
brew unlink python@2
brew link --overwrite python

python3 -m pip install --upgrade pip
