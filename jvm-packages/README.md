# XGBoost4J: Distributed XGBoost for Scala/Java
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org/en/latest/jvm/index.html)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

[Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) |
[Resources](../demo/README.md) |
[Release Notes](../NEWS.md)

XGBoost4J is the JVM package of xgboost. It brings all the optimizations
and power xgboost into JVM ecosystem.

- Train XGBoost models in scala and java with easy customizations.
- Run distributed xgboost natively on jvm frameworks such as
Apache Flink and Apache Spark.

You can find more about XGBoost on [Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) and [Resource Page](../demo/README.md).

## Add Maven Dependency

XGBoost4J, XGBoost4J-Spark, etc. in maven repository is compiled with g++-4.8.5  

### Access SNAPSHOT version

You need to add github as repo:

<b>maven</b>:

```xml
<repository>
  <id>GitHub Repo</id>
  <name>GitHub Repo</name>
  <url>https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/</url>
</repository>
```

<b>sbt</b>:
 
```sbt 
resolvers += "GitHub Repo" at "https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/"
```

the add dependency as following:

<b>maven</b> 

```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j</artifactId>
    <version>latest_version_num</version>
</dependency>
``` 
 
<b>sbt</b> 
```sbt
 "ml.dmlc" % "xgboost4j" % "latest_version_num"
``` 

if you want to use `xgboost4j-spark`, you just need to replace xgboost4j with `xgboost4j-spark`

## Examples

latest_version_num=0.72

Full code examples for Scala, Java, Apache Spark, and Apache Flink can
be found in the [examples package](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example).

**NOTE on LIBSVM Format**: 

* Use *1-based* ascending indexes for the LIBSVM format in distributed training mode

    * Spark does the internal conversion, and does not accept formats that are 0-based

* Whereas, use *0-based* indexes format when predicting in normal mode - for instance, while using the saved model in the Python package
