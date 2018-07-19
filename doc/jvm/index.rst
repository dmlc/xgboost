###################
XGBoost JVM Package
###################

.. raw:: html

  <a href="https://travis-ci.org/dmlc/xgboost">
  <img alt="Build Status" src="https://travis-ci.org/dmlc/xgboost.svg?branch=master">
  </a>
  <a href="https://github.com/dmlc/xgboost/blob/master/LICENSE">
  <img alt="GitHub license" src="http://dmlc.github.io/img/apache2.svg">
  </a>

You have found the XGBoost JVM Package!

************
Installation
************

Installation from source
========================

Building XGBoost4J using Maven requires Maven 3 or newer, Java 7+ and CMake 3.2+ for compiling the JNI bindings.

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

Installation from maven repo
============================

Access release version
----------------------

.. code-block:: xml
  :caption: maven

  <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j</artifactId>
      <version>latest_version_num</version>
  </dependency>

.. code-block:: scala
  :caption: sbt

  "ml.dmlc" % "xgboost4j" % "latest_version_num"

For the latest release version number, please check `here <https://github.com/dmlc/xgboost/releases>`_.

if you want to use XGBoost4J-Spark, you just need to replace ``xgboost4j`` with ``xgboost4j-spark``.

Access SNAPSHOT version
-----------------------

You need to add GitHub as repo:

.. code-block:: xml
  :caption: maven

  <repository>
    <id>GitHub Repo</id>
    <name>GitHub Repo</name>
    <url>https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/</url>
  </repository>

.. code-block:: scala
  :caption: sbt

  resolvers += "GitHub Repo" at "https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/"

Then add dependency as following:

.. code-block:: xml
  :caption: maven

  <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j</artifactId>
      <version>latest_version_num</version>
  </dependency>

.. code-block:: scala
  :caption: sbt

  "ml.dmlc" % "xgboost4j" % "latest_version_num"

For the latest release version number, please check `here <https://github.com/CodingCat/xgboost/tree/maven-repo/ml/dmlc/xgboost4j>`_.

if you want to use XGBoost4J-Spark, you just need to replace ``xgboost4j`` with ``xgboost4j-spark``.

After integrating with Dataframe/Dataset APIs of Spark 2.0, XGBoost4J-Spark only supports compile with Spark 2.x. You can build XGBoost4J-Spark as a component of XGBoost4J by running ``mvn package``, and you can specify the version of spark with ``mvn -Dspark.version=2.0.0 package``.   (To continue working with Spark 1.x, the users are supposed to update pom.xml by modifying the properties like ``spark.version``, ``scala.version``, and ``scala.binary.version``. Users also need to change the implementation by replacing ``SparkSession`` with ``SQLContext`` and the type of API parameters from ``Dataset[_]`` to ``Dataframe``)

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

  Java Overview Tutorial <java_intro>
  Code Examples <https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example>
  Java API doc <http://dmlc.ml/docs/javadocs/index.html>
  XGBoost4J API doc <http://dmlc.ml/docs/scaladocs/xgboost4j/index.html>
  XGBoost4J-Spark API doc <http://dmlc.ml/docs/scaladocs/xgboost4j-spark/index.html>
  XGBoost4J-Flink API doc <http://dmlc.ml/docs/scaladocs/xgboost4j-flink/index.html>
