#!/usr/bin/env bash

set -e

# install gcc/g++ 4.8.2 from tru/devtools-2
sudo apt-get install git openjdk-8-jdk maven

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# build xgboost
cd /xgboost/jvm-packages;mvn package

