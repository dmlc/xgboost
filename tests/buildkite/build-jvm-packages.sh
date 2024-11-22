#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Build and test XGBoost JVM packages with Scala 2.12"
tests/ci_build/ci_build.sh jvm tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION}

echo "--- Build and test XGBoost JVM packages with Scala 2.13"

tests/ci_build/ci_build.sh jvm tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION} "" "" "true"
