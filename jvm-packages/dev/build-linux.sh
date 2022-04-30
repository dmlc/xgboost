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

exec docker run \
  -it \
  --rm  \
  --memory 12g \
  --env JAVA_OPTS="-Xmx9g" \
  --env MAVEN_OPTS="-Xmx3g -Dmaven.repo.local=/xgboost/jvm-packages/dev/.m2" \
  --env CI_BUILD_UID=`id -u` \
  --env CI_BUILD_GID=`id -g` \
  --env CI_BUILD_USER=`id -un` \
  --env CI_BUILD_GROUP=`id -gn` \
  --ulimit core=-1 \
  --volume "${BASEDIR}/../..":/xgboost \
  dmlc/xgboost4j-build \
  /xgboost/tests/ci_build/entrypoint.sh jvm-packages/dev/package-linux.sh "$@"

# CI_BUILD_UID, CI_BUILD_GID, CI_BUILD_USER, CI_BUILD_GROUP
# are used by entrypoint.sh to create the user with the same uid in a container
# so all produced artifacts would be owned by your host user