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

XGBoost4J, XGBoost4J-Spark, etc. in maven repository is compiled with g++-4.8.5.

### Access release version

<b>Maven</b>

```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.12</artifactId>
    <version>latest_version_num</version>
</dependency>
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j-spark_2.12</artifactId>
    <version>latest_version_num</version>
</dependency>
```
or 
```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.13</artifactId>
    <version>latest_version_num</version>
</dependency>
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j-spark_2.13</artifactId>
    <version>latest_version_num</version>
</dependency>
```

<b>sbt</b>
```sbt
libraryDependencies ++= Seq(
  "ml.dmlc" %% "xgboost4j" % "latest_version_num",
  "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num"
)
```

For the latest release version number, please check [here](https://github.com/dmlc/xgboost/releases).


### Access SNAPSHOT version

First add the following Maven repository hosted by the XGBoost project:

<b>Maven</b>:

```xml
<repository>
  <id>XGBoost4J Snapshot Repo</id>
  <name>XGBoost4J Snapshot Repo</name>
  <url>https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/snapshot/</url>
</repository>
```

<b>sbt</b>:

```sbt
resolvers += "XGBoost4J Snapshot Repo" at "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/snapshot/"
```

Then add XGBoost4J as a dependency:

<b>Maven</b>

```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.12</artifactId>
    <version>latest_version_num-SNAPSHOT</version>
</dependency>
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j-spark_2.12</artifactId>
    <version>latest_version_num-SNAPSHOT</version>
</dependency>
```
or with scala 2.13 
```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.13</artifactId>
    <version>latest_version_num-SNAPSHOT</version>
</dependency>
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j-spark_2.13</artifactId>
    <version>latest_version_num-SNAPSHOT</version>
</dependency>
```

<b>sbt</b>
```sbt
libraryDependencies ++= Seq(
  "ml.dmlc" %% "xgboost4j" % "latest_version_num-SNAPSHOT",
  "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num-SNAPSHOT"
)
```

For the latest release version number, please check [the repository listing](https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/list.html).

### GPU algorithm
To enable the GPU algorithm (`tree_method='gpu_hist'`), use artifacts `xgboost4j-gpu_2.12` and `xgboost4j-spark-gpu_2.12` instead.
Note that scala 2.13 is not supported by the [NVIDIA/spark-rapids#1525](https://github.com/NVIDIA/spark-rapids/issues/1525) yet, so the GPU algorithm can only be used with scala 2.12.

## Examples

Full code examples for Scala, Java, Apache Spark, and Apache Flink can
be found in the [examples package](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example).

**NOTE on LIBSVM Format**:

There is an inconsistent issue between XGBoost4J-Spark and other language bindings of XGBoost.

When users use Spark to load trainingset/testset in LIBSVM format with the following code snippet:

```scala
spark.read.format("libsvm").load("trainingset_libsvm")
```

Spark assumes that the dataset is 1-based indexed. However, when you do prediction with other bindings of XGBoost (e.g. Python API of XGBoost), XGBoost assumes that the dataset is 0-based indexed. It creates a pitfall for the users who train model with Spark but predict with the dataset in the same format in other bindings of XGBoost.

## Development

You can build/package xgboost4j locally with the following steps:

**Linux:**
1. Ensure [Docker for Linux](https://docs.docker.com/install/) is installed.
2. Clone this repo: `git clone --recursive https://github.com/dmlc/xgboost.git`
3. Run the following command:
  - With Tests: `./xgboost/jvm-packages/dev/build-linux.sh`
  - Skip Tests: `./xgboost/jvm-packages/dev/build-linux.sh --skip-tests`

**Windows:**
1. Ensure [Docker for Windows](https://docs.docker.com/docker-for-windows/install/) is installed.
2. Clone this repo: `git clone --recursive https://github.com/dmlc/xgboost.git`
3. Run the following command:
  - With Tests: `.\xgboost\jvm-packages\dev\build-linux.cmd`
  - Skip Tests: `.\xgboost\jvm-packages\dev\build-linux.cmd --skip-tests`

*Note: this will create jars for deployment on Linux machines.*
