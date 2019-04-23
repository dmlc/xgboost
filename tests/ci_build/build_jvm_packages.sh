#!/bin/bash

set -e
set -x

# Initialize Maven cache
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages

mvn package -DskipTests

set +x
set +e
