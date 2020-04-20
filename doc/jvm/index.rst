###################
XGBoost JVM Package
###################

.. raw:: html

  <a href="https://travis-ci.org/dmlc/xgboost">
  <img alt="Build Status" src="https://travis-ci.org/dmlc/xgboost.svg?branch=master">
  </a>
  <a href="https://github.com/dmlc/xgboost/blob/master/LICENSE">
  <img alt="GitHub license" src="https://dmlc.github.io/img/apache2.svg">
  </a>

You have found the XGBoost JVM Package!

.. _install_jvm_packages:

************
Installation
************

.. contents::
  :local:
  :backlinks: none

Installation from Maven repository
==================================

Access release version
----------------------
You can use XGBoost4J in your Java/Scala application by adding XGBoost4J as a dependency:

.. code-block:: xml
  :caption: Maven

  <properties>
    ...
    <!-- Specify Scala version in package name -->
    <scala.binary.version>2.12</scala.binary.version>
  </properties>

  <dependencies>
    ...
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j" % "latest_version_num",
    "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num"
  )

This will check out the latest stable version from the Maven Central.

For the latest release version number, please check `here <https://github.com/dmlc/xgboost/releases>`_.

.. note:: Using Maven repository hosted by the XGBoost project

  There may be some delay until a new release becomes available to Maven Central. If you would like to access the latest release immediately, add the Maven repository hosted by the XGBoost project:

  .. code-block:: xml
    :caption: Maven

    <repository>
      <id>XGBoost4J Release Repo</id>
      <name>XGBoost4J Release Repo</name>
      <url>https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/release/</url>
    </repository>

  .. code-block:: scala
    :caption: sbt

    resolvers += "XGBoost4J Release Repo" at "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/release/"

Access SNAPSHOT version
-----------------------

First add the following Maven repository hosted by the XGBoost project:

.. code-block:: xml
  :caption: Maven

  <repository>
    <id>XGBoost4J Snapshot Repo</id>
    <name>XGBoost4J Snapshot Repo</name>
    <url>https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/snapshot/</url>
  </repository>

.. code-block:: scala
  :caption: sbt

  resolvers += "XGBoost4J Snapshot Repo" at "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/snapshot/"

Then add XGBoost4J as a dependency:

.. code-block:: xml
  :caption: maven

  <properties>
    ...
    <!-- Specify Scala version in package name -->
    <scala.binary.version>2.12</scala.binary.version>
  </properties>

  <dependencies>
    ...
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j" % "latest_version_num-SNAPSHOT",
    "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num-SNAPSHOT"
  )

Look up the ``version`` field in `pom.xml <https://github.com/dmlc/xgboost/blob/master/jvm-packages/pom.xml>`_ to get the correct version number.

The SNAPSHOT JARs are hosted by the XGBoost project. Every commit in the ``master`` branch will automatically trigger generation of a new SNAPSHOT JAR. You can control how often Maven should upgrade your SNAPSHOT installation by specifying ``updatePolicy``. See `here <http://maven.apache.org/pom.html#Repositories>`_ for details.

You can browse the file listing of the Maven repository at https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/list.html.

.. note:: Windows not supported by published JARs

  The published JARs from the Maven Central and GitHub currently only supports Linux and MacOS. Windows users should consider building XGBoost4J / XGBoost4J-Spark from the source. Alternatively, checkout pre-built JARs from `criteo-forks/xgboost-jars <https://github.com/criteo-forks/xgboost-jars>`_.

Installation from source
========================

Building XGBoost4J using Maven requires Maven 3 or newer, Java 7+ and CMake 3.3+ for compiling the JNI bindings.

Before you install XGBoost4J, you need to define environment variable ``JAVA_HOME`` as your JDK directory to ensure that your compiler can find ``jni.h`` correctly, since XGBoost4J relies on JNI to implement the interaction between the JVM and native libraries.

After your ``JAVA_HOME`` is defined correctly, it is as simple as run ``mvn package`` under jvm-packages directory to install XGBoost4J. You can also skip the tests by running ``mvn -DskipTests=true package``, if you are sure about the correctness of your local setup.

To publish the artifacts to your local maven repository, run

.. code-block:: bash

  mvn install

Or, if you would like to skip tests, run

.. code-block:: bash

  mvn -DskipTests install

This command will publish the xgboost binaries, the compiled java classes as well as the java sources to your local repository. Then you can use XGBoost4J in your Java projects by including the following dependency in ``pom.xml``:

.. code-block:: xml

  <dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j</artifactId>
    <version>latest_source_version_num</version>
  </dependency>

For sbt, please add the repository and dependency in build.sbt as following:

.. code-block:: scala

  resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

  "ml.dmlc" % "xgboost4j" % "latest_source_version_num"

If you want to use XGBoost4J-Spark, replace ``xgboost4j`` with ``xgboost4j-spark``.

.. note:: XGBoost4J-Spark requires Apache Spark 2.3+

  XGBoost4J-Spark now requires **Apache Spark 2.3+**. Latest versions of XGBoost4J-Spark uses facilities of `org.apache.spark.ml.param.shared` extensively to provide for a tight integration with Spark MLLIB framework, and these facilities are not fully available on earlier versions of Spark.

  Also, make sure to install Spark directly from `Apache website <https://spark.apache.org/>`_. **Upstream XGBoost is not guaranteed to work with third-party distributions of Spark, such as Cloudera Spark.** Consult appropriate third parties to obtain their distribution of XGBoost.

Enabling OpenMP for Mac OS
--------------------------
If you are on Mac OS and using a compiler that supports OpenMP, you need to go to the file ``xgboost/jvm-packages/create_jni.py`` and comment out the line

.. code-block:: python

  CONFIG["USE_OPENMP"] = "OFF"

in order to get the benefit of multi-threading.

********
Contents
********

.. toctree::
  :maxdepth: 2

  java_intro
  XGBoost4J-Spark Tutorial <xgboost4j_spark_tutorial>
  Code Examples <https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example>
  XGBoost4J Java API <javadocs/index>
  XGBoost4J Scala API <scaladocs/xgboost4j/index>
  XGBoost4J-Spark Scala API <scaladocs/xgboost4j-spark/index>
  XGBoost4J-Flink Scala API <scaladocs/xgboost4j-flink/index>
