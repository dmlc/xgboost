#!/usr/bin/env bash

set -x

export JAVA_HOME=/usr/lib/jvm/java-1.8.0
export MAVEN_OPTS="-Xmx3000m"
export CMAKE_CXX_COMPILER=/opt/rh/devtoolset-2/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-2/root/usr/bin/g++
export CC=/opt/rh/devtoolset-2/root/usr/bin/gcc

export PATH=$CXX:$CC:/opt/rh/python27/root/usr/bin/python:$PATH

scl enable devtoolset-2 bash
scl enable python27 bash

rm /usr/bin/python
ln -s /opt/rh/python27/root/usr/bin/python /usr/bin/python

# build xgboost
cd /xgboost/jvm-packages;mvn package

