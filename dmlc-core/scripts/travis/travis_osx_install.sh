#!/bin/bash

if [ ${TRAVIS_OS_NAME} != "osx" ]; then
    exit 0
fi

brew update
