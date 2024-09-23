##################
Installation Guide
##################

XGBoost provides binary packages for some language bindings.  The binary packages support
the GPU algorithm (``device=cuda:0``) on machines with NVIDIA GPUs. Please note that
**training with multiple GPUs is only supported for Linux platform**. See
:doc:`gpu/index`.  Also we have both stable releases and nightly builds, see below for how
to install them.  For building from source, visit :doc:`this page </build>`.

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

.. note:: Parts of the Python package now require glibc 2.28+

  Starting from 2.1.0, XGBoost Python package will be distributed in two variants:

  * ``manylinux_2_28``: for recent Linux distros with glibc 2.28 or newer. This variant comes with all features enabled.
  * ``manylinux2014``: for old Linux distros with glibc older than 2.28. This variant does not support GPU algorithms or federated learning.

  The ``pip`` package manager will automatically choose the correct variant depending on your system.

  Starting from **May 31, 2025**, we will stop distributing the ``manylinux2014`` variant and exclusively
  distribute the ``manylinux_2_28`` variant. We made this decision so that our CI/CD pipeline won't have
  depend on software components that reached end-of-life (such as CentOS 7). We strongly encourage
  everyone to migrate to recent Linux distros in order to use future versions of XGBoost.

  Note. If you want to use GPU algorithms or federated learning on an older Linux distro, you have
  two alternatives:

  1. Upgrade to a recent Linux distro with glibc 2.28+.  OR
  2. Build XGBoost from the source.

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

Minimal installation (CPU-only)
*******************************
The default installation with ``pip`` will install the full XGBoost package, including the support for the GPU algorithms and federated learning.

You may choose to reduce the size of the installed package and save the disk space, by opting to install ``xgboost-cpu`` instead:

.. code-block:: bash

  pip install xgboost-cpu

The ``xgboost-cpu`` variant will have drastically smaller disk footprint, but does not provide some features, such as the GPU algorithms and
federated learning.

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

To force the installation of the GPU variant on a machine that does not have an NVIDIA GPU, use environment variable ``CONDA_OVERRIDE_CUDA``,
as described in `"Managing Virtual Packages" in the conda docs <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html>`_.

.. code-block:: bash

  export CONDA_OVERRIDE_CUDA="12.5"
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

* XGBoost4j-Spark

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
        <artifactId>xgboost4j-spark_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num"
  )

* XGBoost4j-Spark-GPU

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
        <artifactId>xgboost4j-spark-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-spark-gpu" % "latest_version_num"
  )

This will check out the latest stable version from the Maven Central.

For the latest release version number, please check `release page <https://github.com/dmlc/xgboost/releases>`_.

To enable the GPU algorithm (``device='cuda'``), use artifacts ``xgboost4j-spark-gpu_2.12`` instead (note the ``gpu`` suffix).


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

Then add XGBoost4J-Spark as a dependency:

.. code-block:: xml
  :caption: maven

  <properties>
    ...
    <!-- Specify Scala version in package name -->
    <scala.binary.version>2.12</scala.binary.version>
  </properties>

  <dependencies>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-spark" % "latest_version_num-SNAPSHOT"
  )

* XGBoost4j-Spark-GPU

.. code-block:: xml
  :caption: maven

  <properties>
    ...
    <!-- Specify Scala version in package name -->
    <scala.binary.version>2.12</scala.binary.version>
  </properties>

  <dependencies>
    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark-gpu_${scala.binary.version}</artifactId>
        <version>latest_version_num-SNAPSHOT</version>
    </dependency>
  </dependencies>

.. code-block:: scala
  :caption: sbt

  libraryDependencies ++= Seq(
    "ml.dmlc" %% "xgboost4j-spark-gpu" % "latest_version_num-SNAPSHOT"
  )


Look up the ``version`` field in `pom.xml <https://github.com/dmlc/xgboost/blob/master/jvm-packages/pom.xml>`_ to get the correct version number.

The SNAPSHOT JARs are hosted by the XGBoost project. Every commit in the ``master`` branch will automatically trigger generation of a new SNAPSHOT JAR. You can control how often Maven should upgrade your SNAPSHOT installation by specifying ``updatePolicy``. See `here <http://maven.apache.org/pom.html#Repositories>`_ for details.

You can browse the file listing of the Maven repository at https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/list.html.

To enable the GPU algorithm (``device='cuda'``), use artifacts ``xgboost4j-gpu_2.12`` and ``xgboost4j-spark-gpu_2.12`` instead (note the ``gpu`` suffix).
