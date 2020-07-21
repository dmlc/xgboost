#!/bin/bash

set -e
set -x


nvidia-smi

ls /usr/local/

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

# Get version number of XGBoost4J and other auxiliary information
cd jvm-packages
xgboost4j_version=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)
scala_binary_version=$(mvn help:evaluate -Dexpression=scala.binary.version -q -DforceStdout)

python3 xgboost4j-tester/get_iris.py
xgb_jars="./xgboost4j/target/xgboost4j_${scala_binary_version}-${xgboost4j_version}.jar,./xgboost4j-spark/target/xgboost4j-spark_${scala_binary_version}-${xgboost4j_version}.jar"
example_jar="./xgboost4j-example/target/xgboost4j-example_${scala_binary_version}-${xgboost4j_version}.jar"

echo "Run SparkTraining locally ... "
spark-submit \
  --master 'local[1]' \
  --class ml.dmlc.xgboost4j.scala.example.spark.SparkTraining \
  --jars $xgb_jars \
    $example_jar \
      ${PWD}/iris.csv gpu \

echo "Run SparkMLlibPipeline locally ... "
spark-submit \
  --master 'local[1]' \
  --class ml.dmlc.xgboost4j.scala.example.spark.SparkMLlibPipeline \
  --jars $xgb_jars \
    $example_jar \
      ${PWD}/iris.csv ${PWD}/native_model ${PWD}/pipeline_model gpu \

set +x
set +e
