##################
Installation Guide
##################

XGBoost provides binary packages for some language bindings.  The binary packages support
the GPU algorithm (``gpu_hist``) on machines with NVIDIA GPUs. Please note that **training
with multiple GPUs is only supported for Linux platform**. See :doc:`gpu/index`.  Also we
have both stable releases and nightly builds, see below for how to install them.  For
building from source, visit :doc:`this page </build>`.

.. contents:: Contents

Stable Release
==============

Python
------

Pre-built binary wheels are uploaded to PyPI (Python Package Index) for each release. Supported platforms are Linux (x86_64, aarch64), Windows (x86_64) and MacOS (x86_64, Apple Silicon).

.. code-block:: bash

  # Pip 21.3+ is required
  pip install xgboost


You might need to run the command with ``--user`` flag or use ``virtualenv`` if you run
into permission errors.

.. note:: Windows users need to install Visual C++ Redistributable

  XGBoost requires DLLs from `Visual C++ Redistributable
  <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_
  in order to function, so make sure to install it. Exception: If
  you have Visual Studio installed, you already have access to
  necessary libraries and thus don't need to install Visual C++
  Redistributable.


Capabilities of binary wheels for each platform:

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+---------------------+---------+----------------------+
| Platform            | GPU     | Multi-Node-Multi-GPU |
+=====================+=========+======================+
| Linux x86_64        | |tick|  |  |tick|              |
+---------------------+---------+----------------------+
| Linux aarch64       | |cross| |  |cross|             |
+---------------------+---------+----------------------+
| MacOS x86_64        | |cross| |  |cross|             |
+---------------------+---------+----------------------+
| MacOS Apple Silicon | |cross| |  |cross|             |
+---------------------+---------+----------------------+
| Windows             | |tick|  |  |cross|             |
+---------------------+---------+----------------------+

Conda
*****

You may use the Conda packaging manager to install XGBoost:

.. code-block:: bash

   conda install -c conda-forge py-xgboost

Conda should be able to detect the existence of a GPU on your machine and install the correct variant of XGBoost. If you run into issues, try indicating the variant explicitly:

.. code-block:: bash

   # CPU only
   conda install -c conda-forge py-xgboost-cpu
   # Use NVIDIA GPU
   conda install -c conda-forge py-xgboost-gpu


Visit the `Miniconda website <https://docs.conda.io/en/latest/miniconda.html>`_ to obtain Conda.

.. note:: ``py-xgboost-gpu`` not available on Windows.

   The ``py-xgboost-gpu`` is currently not available on Windows. If you are using Windows,
   please use ``pip`` to install XGBoost with GPU support.

R
-

* From CRAN:

  .. code-block:: R

    install.packages("xgboost")

  .. note:: Using all CPU cores (threads) on Mac OSX

     If you are using Mac OSX, you should first install OpenMP library (``libomp``) by running

     .. code-block:: bash

        brew install libomp

     and then run ``install.packages("xgboost")``. Without OpenMP, XGBoost will only use a
     single CPU core, leading to suboptimal training speed.

* We also provide **experimental** pre-built binary with GPU support. With this binary,
  you will be able to use the GPU algorithm without building XGBoost from the source.
  Download the binary package from the Releases page. The file name will be of the form
  ``xgboost_r_gpu_[os]_[version].tar.gz``, where ``[os]`` is either ``linux`` or ``win64``.
  (We build the binaries for 64-bit Linux and Windows.)
  Then install XGBoost by running:

  .. code-block:: bash

    # Install dependencies
    R -q -e "install.packages(c('data.table', 'jsonlite'))"
    # Install XGBoost
    R CMD INSTALL ./xgboost_r_gpu_linux.tar.gz

JVM
---

* XGBoost4j/XGBoost4j-Spark

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

* XGBoost4j-GPU/XGBoost4j-Spark-GPU

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
        <artifactId>xgboost4j-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-gpu" % "latest_version_num",
    "ml.dmlc" %% "xgboost4j-spark-gpu" % "latest_version_num"
  )

This will check out the latest stable version from the Maven Central.

For the latest release version number, please check `release page <https://github.com/dmlc/xgboost/releases>`_.

To enable the GPU algorithm (``tree_method='gpu_hist'``), use artifacts ``xgboost4j-gpu_2.12`` and ``xgboost4j-spark-gpu_2.12`` instead (note the ``gpu`` suffix).


.. note:: Windows not supported in the JVM package

  Currently, XGBoost4J-Spark does not support Windows platform, as the distributed training algorithm is inoperational for Windows. Please use Linux or MacOS.


Nightly Build
=============


Python
------

Nightly builds are available. You can go to `this page <https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html>`_,
find the wheel with the commit ID you want and install it with pip:

.. code-block:: bash

  pip install <url to the wheel>


The capability of Python pre-built wheel is the same as stable release.


R
-

Other than standard CRAN installation, we also provide *experimental* pre-built binary on
with GPU support.  You can go to `this page
<https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html>`_, Find the commit
ID you want to install and then locate the file ``xgboost_r_gpu_[os]_[commit].tar.gz``,
where ``[os]`` is either ``linux`` or ``win64``. (We build the binaries for 64-bit Linux
and Windows.) Download it and run the following commands:

.. code-block:: bash

  # Install dependencies
  R -q -e "install.packages(c('data.table', 'jsonlite', 'remotes'))"
  # Install XGBoost
  R CMD INSTALL ./xgboost_r_gpu_linux.tar.gz


JVM
---

* XGBoost4j/XGBoost4j-Spark

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

* XGBoost4j-GPU/XGBoost4j-Spark-GPU

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
        <artifactId>xgboost4j-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-gpu" % "latest_version_num-SNAPSHOT",
    "ml.dmlc" %% "xgboost4j-spark-gpu" % "latest_version_num-SNAPSHOT"
  )


Look up the ``version`` field in `pom.xml <https://github.com/dmlc/xgboost/blob/master/jvm-packages/pom.xml>`_ to get the correct version number.

The SNAPSHOT JARs are hosted by the XGBoost project. Every commit in the ``master`` branch will automatically trigger generation of a new SNAPSHOT JAR. You can control how often Maven should upgrade your SNAPSHOT installation by specifying ``updatePolicy``. See `here <http://maven.apache.org/pom.html#Repositories>`_ for details.

You can browse the file listing of the Maven repository at https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/list.html.

To enable the GPU algorithm (``tree_method='gpu_hist'``), use artifacts ``xgboost4j-gpu_2.12`` and ``xgboost4j-spark-gpu_2.12`` instead (note the ``gpu`` suffix).
