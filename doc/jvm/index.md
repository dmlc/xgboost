XGBoost JVM Package
===================
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

You have find XGBoost JVM Package!

Installation
------------
Currently, XGBoost4J only support installation from source. Building XGBoost4J using Maven requires Maven 3 or newer and Java 7+.

Before you install XGBoost4J, you need to define environment variable `JAVA_HOME` as your JDK directory to ensure that your compiler can find `jni.h` correctly, since XGBoost4J relies on JNI to implement the interaction between the JVM and native libraries.

After your `JAVA_HOME` is defined correctly, it is as simple as run `mvn package` under jvm-packages directory to install XGBoost4J. You can also skip the tests by running `mvn -DskipTests=true package`, if you are sure about the correctness of your local setup.

After integrating with Dataframe/Dataset APIs of Spark 2.0, XGBoost4J-Spark only supports compile with Spark 2.x. You can build XGBoost4J-Spark as a component of XGBoost4J by running `mvn package`, and you can specify the version of spark with `mvn -Dspark.version=1.6.0 package`. (In the `dataframe` sub directory under `xgboost4j-spark`, we provide the dataframe implementation based on Spark-1.6.x APIs, but this folder will be removed in future release)

Contents
--------
* [Java Overview Tutorial](java_intro.md)

Resources
---------
* [Code Examples](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example)
* [Java API Docs](http://dmlc.ml/docs/javadocs/index.html)

## Scala API Docs
  * [XGBoost4J](http://dmlc.ml/docs/scaladocs/xgboost4j/index.html)
  * [XGBoost4J-Spark](http://dmlc.ml/docs/scaladocs/xgboost4j-spark/index.html)
  * [XGBoost4J-Flink](http://dmlc.ml/docs/scaladocs/xgboost4j-flink/index.html)
