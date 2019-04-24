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
scala_version=$(mvn help:evaluate -Dexpression=scala.version -q -DforceStdout)
scala_binary_version=$(mvn help:evaluate -Dexpression=scala.binary.version -q -DforceStdout)

# Install XGBoost4J JAR into local Maven repository
mvn install:install-file -Dfile=./xgboost4j/target/xgboost4j-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j -Dversion=${xgboost4j_version} -Dpackaging=jar
mvn install:install-file -Dfile=./xgboost4j/target/xgboost4j-${xgboost4j_version}-tests.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j -Dversion=${xgboost4j_version} -Dpackaging=test-jar -Dclassifier=tests

cd xgboost4j-tester
# Generate pom.xml for XGBoost4J-tester, a dummy project to run XGBoost4J tests
python3 ./generate_pom.py ${xgboost4j_version} ${maven_compiler_source} ${maven_compiler_target} ${scala_version} ${scala_binary_version}
# Run tests
mvn test

set +x
set +e
