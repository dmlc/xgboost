XGBoost JVM Package
===================
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

You have find XGBoost JVM Package!

Installation
------------
To build XGBoost4J contains two steps.
- First type the following command to build JNI library.
```bash
./create_jni.sh
```
- Then package the libary. you can run `mvn package` in xgboost4j folder or just use IDE(eclipse/netbeans) to open this maven project and build.

Contents
--------
* [Java Overview Tutorial](java_intro.md)
* [Code Examples](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example)
* [Java API Docs](http://dmlc.ml/docs/javadocs/index.html)
* [Scala API Docs]
  * [XGBoost4J](http://dmlc.ml/docs/scaladocs/xgboost4j/index.html)
  * [XGBoost4J-Spark](http://dmlc.ml/docs/scaladocs/xgboost4j-spark/index.html)
  * [XGBoost4J-Flink](http://dmlc.ml/docs/scaladocs/xgboost4j-flink/index.html)