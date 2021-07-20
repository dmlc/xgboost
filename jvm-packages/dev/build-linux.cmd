@echo off

rem
rem Licensed to the Apache Software Foundation (ASF) under one
rem or more contributor license agreements.  See the NOTICE file
rem distributed with this work for additional information
rem regarding copyright ownership.  The ASF licenses this file
rem to you under the Apache License, Version 2.0 (the
rem "License"); you may not use this file except in compliance
rem with the License.  You may obtain a copy of the License at
rem
rem   http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing,
rem software distributed under the License is distributed on an
rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
rem KIND, either express or implied.  See the License for the
rem specific language governing permissions and limitations
rem under the License.
rem

rem The the local path of this file
set "BASEDIR=%~dp0"

rem The local path of .m2 directory for maven
set "M2DIR=%BASEDIR%\.m2\"

rem Create a local .m2 directory if needed
if not exist "%M2DIR%" mkdir "%M2DIR%"

rem Build and tag the Dockerfile
docker build -t dmlc/xgboost4j-build %BASEDIR%

docker run^
 -it^
 --rm^
 --memory 12g^
 --env JAVA_OPTS="-Xmx9g"^
 --env MAVEN_OPTS="-Xmx3g"^
 --ulimit core=-1^
 --volume %BASEDIR%\..\..:/xgboost^
 --volume %M2DIR%:/root/.m2^
 dmlc/xgboost4j-build^
 /xgboost/jvm-packages/dev/package-linux.sh "%*"
