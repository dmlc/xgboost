#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Build XGBoost JVM packages scala 2.12"
tests/ci_build/ci_build.sh jvm tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION}

echo "--- Stash XGBoost4J JARs (Scala 2.12)"
buildkite-agent artifact upload "jvm-packages/xgboost4j/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-spark/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-flink/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-example/target/*.jar"

echo "--- Build XGBoost JVM packages scala 2.13"

tests/ci_build/ci_build.sh jvm tests/ci_build/build_jvm_packages.sh \
  ${SPARK_VERSION} "" "" "true"

echo "--- Stash XGBoost4J JARs (Scala 2.13)"
buildkite-agent artifact upload "jvm-packages/xgboost4j/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-spark/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-flink/target/*.jar"
buildkite-agent artifact upload "jvm-packages/xgboost4j-example/target/*.jar"
