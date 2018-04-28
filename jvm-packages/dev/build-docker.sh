#!/bin/bash

set -x

sudo docker run --rm -m 4g -e JAVA_OPTS='-Xmx6g' --attach stdin --attach stdout --attach stderr --volume `pwd`/../:/xgboost codingcat/xgboost:latest /xgboost/jvm-packages/dev/build.sh
