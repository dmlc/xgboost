#!/bin/bash

set -e
set -x

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

# Get version number of XGBoost4J and other auxiliary information
cd jvm-packages
xgboost4j_version=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)
maven_compiler_source=$(mvn help:evaluate -Dexpression=maven.compiler.source -q -DforceStdout)
maven_compiler_target=$(mvn help:evaluate -Dexpression=maven.compiler.target -q -DforceStdout)
spark_version=$(mvn help:evaluate -Dexpression=spark.version -q -DforceStdout)
scala_version=$(mvn help:evaluate -Dexpression=scala.version -q -DforceStdout)
scala_binary_version=$(mvn help:evaluate -Dexpression=scala.binary.version -q -DforceStdout)

# Install XGBoost4J JAR into local Maven repository
mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j/target/xgboost4j_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar
mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j/target/xgboost4j_${scala_binary_version}-${xgboost4j_version}-tests.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=test-jar -Dclassifier=tests
mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j-spark/target/xgboost4j-spark_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j-spark_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar
mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j-example/target/xgboost4j-example_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j-example_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar

cd xgboost4j-tester
# Generate pom.xml for XGBoost4J-tester, a dummy project to run XGBoost4J tests
python3 ./generate_pom.py ${xgboost4j_version} ${maven_compiler_source} ${maven_compiler_target} ${spark_version} ${scala_version} ${scala_binary_version}
# Run unit tests with XGBoost4J
mvn --no-transfer-progress package

# Run integration tests with XGBoost4J
java -jar ./target/xgboost4j-tester_${scala_binary_version}-1.0-SNAPSHOT-jar-with-dependencies.jar

# Run integration tests with XGBoost4J-Spark
if [ ! -z "$RUN_INTEGRATION_TEST" ]
then
  python3 get_iris.py
  spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkTraining --master 'local[8]' ./target/xgboost4j-tester_${scala_binary_version}-1.0-SNAPSHOT-jar-with-dependencies.jar ${PWD}/iris.csv
  spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkMLlibPipeline --master 'local[8]' ./target/xgboost4j-tester_${scala_binary_version}-1.0-SNAPSHOT-jar-with-dependencies.jar ${PWD}/iris.csv ${PWD}/native_model ${PWD}/pipeline_model
fi

set +x
set +e
