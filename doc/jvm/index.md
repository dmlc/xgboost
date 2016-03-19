XGBoost JVM Package
===================
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

You have find XGBoost JVM Package!

Installation
------------
Currently, XGBoost4J only support installation from source. Building XGBoost4J using Maven requires Maven 3 or newer and Java 7+. 

Before you install XGBoost4J, you need to define environment variable `JAVA_HOME` as your JDK directory to ensure that your compiler can find `jni.h` correctly, since XGBoost4J relies on JNI to implement the interaction between the JVM and native libraries.

After your `JAVA_HOME` is defined correctly, it is as simple as run `mvn package` under jvm-packages directory to install XGBoost4J.

Contents
--------
* [Java Overview Tutorial](java_intro.md)
* [Code Examples](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example)
* [Java API Docs](http://dmlc.ml/docs/javadocs/index.html)
* [Scala API Docs]
  * [XGBoost4J](http://dmlc.ml/docs/scaladocs/xgboost4j/index.html)
  * [XGBoost4J-Spark](http://dmlc.ml/docs/scaladocs/xgboost4j-spark/index.html)
  * [XGBoost4J-Flink](http://dmlc.ml/docs/scaladocs/xgboost4j-flink/index.html)