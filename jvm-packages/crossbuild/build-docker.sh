#!/bin/bash

set -x

DOCKER_LINUX_X64_CONTAINER=`docker ps -aqf name=xgboost_linux_x64-be`; 

if [ -z "$DOCKER_LINUX_X64_CONTAINER" ]; then \
		docker container create --attach stdin --attach stdout --attach stderr --volume `pwd`:/xgboost --name xgboost_linux_x64 codingcat/xgboost:latest /xgboost/jvm-packages/crossbuild/build.sh; 
fi

docker start -a xgboost_linux_x64
