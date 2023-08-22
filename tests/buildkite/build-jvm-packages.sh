#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Build XGBoost JVM packages scala 2.12"
tests/ci_build/ci_build.sh jvm docker tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION}


echo "--- Build XGBoost JVM packages scala 2.13"

tests/ci_build/ci_build.sh jvm docker tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION} "" "" "true"

echo "--- Stash XGBoost4J JARs"
buildkite-agent artifact upload "jvm-packages/xgboost4j/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-spark/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-flink/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-example/target/*.jar"
