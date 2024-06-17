#!/bin/bash

set -e
set -x

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

cd jvm-packages
jvm_packages_dir=`pwd`
# Get version number of XGBoost4J and other auxiliary information
xgboost4j_version=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)
maven_compiler_source=$(mvn help:evaluate -Dexpression=maven.compiler.source -q -DforceStdout)
maven_compiler_target=$(mvn help:evaluate -Dexpression=maven.compiler.target -q -DforceStdout)
spark_version=$(mvn help:evaluate -Dexpression=spark.version -q -DforceStdout)

if [ ! -z "$RUN_INTEGRATION_TEST" ]; then
  cd $jvm_packages_dir/xgboost4j-tester
  python3 get_iris.py
  cd $jvm_packages_dir
fi

for scala_binary_version in "2.12" "2.13"; do
  cd ..
  python dev/change_scala_version.py --scala-version ${scala_binary_version}
  cd jvm-packages
  scala_version=$(mvn help:evaluate -Dexpression=scala.version -q -DforceStdout)

  # Install XGBoost4J JAR into local Maven repository
  mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j/target/xgboost4j_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar -DgeneratePom=true
  mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j/target/xgboost4j_${scala_binary_version}-${xgboost4j_version}-tests.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=test-jar -Dclassifier=tests -DgeneratePom=true
  mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j-spark/target/xgboost4j-spark_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j-spark_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar -DgeneratePom=true
  mvn --no-transfer-progress install:install-file -Dfile=./xgboost4j-example/target/xgboost4j-example_${scala_binary_version}-${xgboost4j_version}.jar -DgroupId=ml.dmlc -DartifactId=xgboost4j-example_${scala_binary_version} -Dversion=${xgboost4j_version} -Dpackaging=jar -DgeneratePom=true

  cd xgboost4j-tester
  # Generate pom.xml for XGBoost4J-tester, a dummy project to run XGBoost4J tests
  python3 ./generate_pom.py ${xgboost4j_version} ${maven_compiler_source} ${maven_compiler_target} ${spark_version} ${scala_version} ${scala_binary_version}
  # Build package and unit tests with XGBoost4J
  mvn --no-transfer-progress clean package
  xgboost4j_tester_jar="$jvm_packages_dir/xgboost4j-tester/target/xgboost4j-tester_${scala_binary_version}-1.0-SNAPSHOT-jar-with-dependencies.jar"
  # Run integration tests with XGBoost4J
  java -jar $xgboost4j_tester_jar

  # Run integration tests with XGBoost4J-Spark
  if [ ! -z "$RUN_INTEGRATION_TEST" ]; then
    # Changing directory so that we do not mix code and resulting files
    cd target
    if [[ "$scala_binary_version" == "2.12" ]]; then
       /opt/spark-scala-2.12/bin/spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkTraining --master 'local[8]' ${xgboost4j_tester_jar} $jvm_packages_dir/xgboost4j-tester/iris.csv
       /opt/spark-scala-2.12/bin/spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkMLlibPipeline --master 'local[8]' ${xgboost4j_tester_jar} $jvm_packages_dir/xgboost4j-tester/iris.csv ${PWD}/native_model-${scala_version} ${PWD}/pipeline_model-${scala_version}
    elif [[ "$scala_binary_version" == "2.13" ]]; then
      /opt/spark-scala-2.13/bin/spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkTraining --master 'local[8]' ${xgboost4j_tester_jar} $jvm_packages_dir/xgboost4j-tester/iris.csv
      /opt/spark-scala-2.13/bin/spark-submit --class ml.dmlc.xgboost4j.scala.example.spark.SparkMLlibPipeline --master 'local[8]' ${xgboost4j_tester_jar} $jvm_packages_dir/xgboost4j-tester/iris.csv ${PWD}/native_model-${scala_version} ${PWD}/pipeline_model-${scala_version}
    else
      echo "Unexpected scala version: $scala_version ($scala_binary_version)."
    fi
  fi
  cd $jvm_packages_dir
done

set +x
set +e
