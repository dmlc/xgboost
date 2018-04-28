#!/usr/bin/env bash

set -e

apt-get update
apt-get -y install g++-4.8
apt-get -y install git openjdk-8-jdk maven python cmake
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export CXX=g++
export MAVEN_OPTS="-Xmx3000m"

# build xgboost
cd /xgboost/jvm-packages;mvn package

