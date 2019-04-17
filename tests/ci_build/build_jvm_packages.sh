#!/bin/bash

set -e
set -x

# Initialize Maven cache
./tests/ci_build/initialize_maven.sh

cd jvm-packages

mvn package

set +x
set +e
