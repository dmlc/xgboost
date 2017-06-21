#!/usr/bin/env bash

set -e

# install gcc/g++ 4.8.2 from tru/devtools-2
apt-get update
apt-get -y dist-upgrade
apt-get -y install build-essential
apt-get -y install git openjdk-8-jdk maven python cmake

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export CXX=g++
export MAVEN_OPTS="-Xmx3000m"

# build xgboost
cd /xgboost/jvm-packages;mvn package

