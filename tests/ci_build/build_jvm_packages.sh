#!/bin/bash

set -e
set -x

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages

mvn --no-transfer-progress package

set +x
set +e
