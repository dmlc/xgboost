#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
BASEDIR="$( cd "$( dirname "$0" )" && pwd )" # the directory of this file

docker build -t dmlc/xgboost4j-build "${BASEDIR}" # build and tag the Dockerfile

docker run \
  -it \
  --rm  \
  --memory 8g \
  --env JAVA_OPTS="-Xmx6g" \
  --env MAVEN_OPTS="-Xmx2g" \
  --ulimit core=-1 \
  --volume "${BASEDIR}/../..":/xgboost \
  --volume "${BASEDIR}/.m2":/root/.m2 \
  dmlc/xgboost4j-build \
  /xgboost/jvm-packages/dev/package-linux.sh "$@"
