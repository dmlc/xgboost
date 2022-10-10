#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Test XGBoost4J on a machine with JDK ${JDK_VERSION}, Spark ${SPARK_VERSION}"
buildkite-agent artifact download "jvm-packages/xgboost4j/target/*.jar" . --step build-jvm-packages
buildkite-agent artifact download "jvm-packages/xgboost4j-spark/target/*.jar" . --step build-jvm-packages
buildkite-agent artifact download "jvm-packages/xgboost4j-example/target/*.jar" . --step build-jvm-packages
export CI_DOCKER_EXTRA_PARAMS_INIT='-e RUN_INTEGRATION_TEST=1'
tests/ci_build/ci_build.sh jvm_cross docker --build-arg JDK_VERSION=${JDK_VERSION} \
  --build-arg SPARK_VERSION=${SPARK_VERSION} tests/ci_build/test_jvm_cross.sh
