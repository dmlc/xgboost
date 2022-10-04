#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Build XGBoost JVM packages"
tests/ci_build/ci_build.sh jvm docker tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION}

echo "--- Stash XGBoost4J JARs"
buildkite-agent artifact upload "jvm-packages/xgboost4j/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-spark/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-example/target/*.jar"
