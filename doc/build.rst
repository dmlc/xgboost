####################
Building From Source
####################

This page gives instructions on how to build and install XGBoost from the source code on
various systems.  If the instructions do not work for you, please feel free to ask
questions at `GitHub <https://github.com/dmlc/xgboost/issues>`__.

.. note:: Pre-built binary is available: now with GPU support

  Consider installing XGBoost from a pre-built binary, to avoid the trouble of building XGBoost from the source.  Checkout :doc:`Installation Guide </install>`.

.. contents:: Contents
  :local:

.. _get_source:

*************************
Obtaining the Source Code
*************************

To obtain the development repository of XGBoost, one needs to use ``git``. XGBoost uses
Git submodules to manage dependencies. So when you clone the repo, remember to specify
``--recursive`` option:

  .. code-block:: bash

    git clone --recursive https://github.com/dmlc/xgboost

.. _build_shared_lib:

***************************
Building the Shared Library
***************************

This section describes the procedure to build the shared library and CLI interface
independently. For building language specific package, see corresponding sections in this
document.

- On Linux and other UNIX-like systems, the target library is ``libxgboost.so``
- On MacOS, the target library is ``libxgboost.dylib``
- On Windows the target library is ``xgboost.dll``

This shared library is used by different language bindings (with some additions depending
on the binding you choose).  The minimal building requirement is

- A recent C++ compiler supporting C++17. We use gcc, clang, and MSVC for daily
  testing. Mingw is only used for the R package and has limited features.
- CMake 3.18 or higher.

For a list of CMake options like GPU support, see ``#-- Options`` in CMakeLists.txt on top
level of source tree. We use ``ninja`` for build in this document, specified via the CMake
flag ``-GNinja``. If you prefer other build tools like ``make`` or ``Visual Studio 17
2022``, please change the corresponding CMake flags. Consult the `CMake generator
<https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html>`_ document when
needed.

.. _running_cmake_and_build:

Running CMake and build
=======================

After obtaining the source code, one builds XGBoost by running CMake:

.. code-block:: bash

  cd xgboost
  cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
  cd build && ninja


The same command applies for both Unix-like systems and Windows. After running the
build, one should see a shared object under the ``xgboost/lib`` directory.

- Building on MacOS

  On MacOS, one needs to obtain ``libomp`` from `Homebrew <https://brew.sh/>`_ first:

  .. code-block:: bash

    brew install libomp

- Visual Studio

  The latest Visual Studio has builtin support for CMake projects. If you prefer using an
  IDE over the command line, you can use the ``open with visual studio`` option in the
  right-click menu under the ``xgboost`` source directory. Consult the VS `document
  <https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170>`__
  for more info.

.. _build_gpu_support:


Building with GPU support
=========================

XGBoost can be built with GPU support for both Linux and Windows using CMake. See
`Building R package with GPU support`_ for special instructions for R.

An up-to-date version of the CUDA toolkit is required.

.. note:: Checking your compiler version

    CUDA is really picky about supported compilers, a table for the compatible compilers
    for the latest CUDA version on Linux can be seen `here
    <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

Some distros package a compatible ``gcc`` version with CUDA. If you run into compiler
errors with ``nvcc``, try specifying the correct compiler with
``-DCMAKE_CXX_COMPILER=/path/to/correct/g++ -DCMAKE_C_COMPILER=/path/to/correct/gcc``. On
Arch Linux, for example, both binaries can be found under ``/opt/cuda/bin/``. In addition,
the ``CMAKE_CUDA_HOST_COMPILER`` parameter can be useful.

From the command line on Linux starting from the XGBoost directory, add the ``USE_CUDA``
flag:

.. code-block:: bash

  cmake -B build -S . -DUSE_CUDA=ON -GNinja
  cd build && ninja

To speed up compilation, the compute version specific to your GPU could be passed to cmake
as, e.g., ``-DCMAKE_CUDA_ARCHITECTURES=75``. A quick explanation and numbers for some
architectures can be found `in this page <https://developer.nvidia.com/cuda-gpus>`_.

- Faster distributed GPU training with NCCL

  By default, distributed GPU training is enabled with the option
  ``USE_NCCL=ON``. Distributed GPU training depends on NCCL2, available at `this link
  <https://developer.nvidia.com/nccl>`_. Since NCCL2 is only available for Linux machines,
  **Distributed GPU training is available only for Linux**.

  .. code-block:: bash

    cmake -B build -S . -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/path/to/nccl2 -GNinja
    cd build && ninja

  Some additional flags are available for NCCL, ``BUILD_WITH_SHARED_NCCL`` enables
  building XGBoost with NCCL as a shared library, while ``USE_DLOPEN_NCCL`` enables
  XGBoost to load NCCL at runtime using ``dlopen``.

Federated Learning
==================

The federated learning plugin requires ``grpc`` and ``protobuf``. To install grpc, refer
to the `installation guide from the gRPC website
<https://grpc.io/docs/languages/cpp/quickstart/>`_. Alternatively, one can use the
``libgrpc`` and the ``protobuf`` package from conda forge if conda is available. After
obtaining the required dependencies, enable the flag: ``-DPLUGIN_FEDERATED=ON`` when
running CMake. Please note that only Linux is supported for the federated plugin.


.. code-block:: bash

  cmake -B build -S . -DPLUGIN_FEDERATED=ON -GNinja
  cd build && ninja


.. _build_python:

***********************************
Building Python Package from Source
***********************************

The Python package is located at ``python-package/``.

Building Python Package with Default Toolchains
===============================================
There are several ways to build and install the package from source:

1. Build C++ core with CMake first

  You can first build C++ library using CMake as described in :ref:`build_shared_lib`.
  After compilation, a shared library will appear in ``lib/`` directory.
  On Linux distributions, the shared library is ``lib/libxgboost.so``.
  The install script ``pip install .`` will reuse the shared library instead of compiling
  it from scratch, making it quite fast to run.

  .. code-block:: console

    $ cd python-package/
    $ pip install .  # Will re-use lib/libxgboost.so

2. Install the Python package directly

  If the shared object is not present, the Python project setup script will try to run the
  CMake build command automatically. Navigate to the ``python-package/`` directory and
  install the Python package by running:

  .. code-block:: console

    $ cd python-package/
    $ pip install -v . # Builds the shared object automatically.

  which will compile XGBoost's native (C++) code using default CMake flags.  To enable
  additional compilation options, pass corresponding ``--config-settings``:

  .. code-block:: console

    $ pip install -v . --config-settings use_cuda=True --config-settings use_nccl=True

  Use Pip 22.1 or later to use ``--config-settings`` option.

  Here are the available options for ``--config-settings``:

  .. literalinclude:: ../python-package/packager/build_config.py
    :language: python
    :start-at: @dataclasses.dataclass
    :end-before: def _set_config_setting(

  ``use_system_libxgboost`` is a special option. See Item 4 below for
  detailed description.

  .. note:: Verbose flag recommended

    As ``pip install .`` will build C++ code, it will take a while to complete.
    To ensure that the build is progressing successfully, we suggest that
    you add the verbose flag (``-v``) when invoking ``pip install``.


3. Editable installation

  To further enable rapid development and iteration, we provide an **editable
  installation**.  In an editable installation, the installed package is simply a symbolic
  link to your working copy of the XGBoost source code. So every changes you make to your
  source directory will be immediately visible to the Python interpreter. To install
  XGBoost as editable installation, first build the shared library as previously described
  in :ref:`running_cmake_and_build`, then install the Python package with the ``-e`` flag:

  .. code-block:: bash

    # Build shared library libxgboost.so
    cmake -B build -S . -GNinja
    cd build && ninja
    # Install as editable installation
    cd ../python-package
    pip install -e .

4. Reuse the ``libxgboost.so`` on system path.

  This option is useful for package managers that wish to separately package
  ``libxgboost.so`` and the XGBoost Python package. For example, Conda
  publishes ``libxgboost`` (for the shared library) and ``py-xgboost``
  (for the Python package).

  To use this option, first make sure that ``libxgboost.so`` exists in the system library path:

  .. code-block:: python

    import sys
    import pathlib
    libpath = pathlib.Path(sys.base_prefix).joinpath("lib", "libxgboost.so")
    assert libpath.exists()

  Then pass ``use_system_libxgboost=True`` option to ``pip install``:

  .. code-block:: bash

    cd python-package
    pip install . --config-settings use_system_libxgboost=True


.. note::

  See :doc:`contrib/python_packaging` for instructions on packaging and distributing
  XGBoost as Python distributions.


******************************
Building R Package From Source
******************************

By default, the package installed by running ``install.packages`` is built from source
using the package from `CRAN <https://cran.r-project.org/>`__.  Here we list some other
options for installing development version.

Installing the development version (Linux / Mac OSX)
====================================================

Make sure you have installed git and a recent C++ compiler supporting C++11 (See above
sections for requirements of building C++ core).

Due to the use of git-submodules, ``remotes::install_github()`` cannot be used to
install the latest version of R package. Thus, one has to run git to check out the code
first, see :ref:`get_source` on how to initialize the git repository for XGBoost. The
simplest way to install the R package after obtaining the source code is:

.. code-block:: bash

  cd R-package
  R CMD INSTALL .

Use the environment variable ``MAKEFLAGS=-j$(nproc)`` if you want to speedup the build. As
an alternative, the package can also be loaded through ``devtools::load_all()`` from the
same subfolder ``R-package`` in the repository's root, and by extension, can be installed
through RStudio's build panel if one adds that folder ``R-package`` as an R package
project in the RStudio IDE.

.. code-block:: R

  library(devtools)
  devtools::load_all(path = "/path/to/xgboost/R-package")

On Linux, if you want to use the CMake build for greater flexibility around compile flags,
the earlier snippet can be replaced by:

.. code-block:: bash

  cmake -B build -S . -DR_LIB=ON -GNinja
  cd build && ninja install

.. warning::

   MSVC is not supported for the R package as it has difficulty handling R C
   headers. CMake build is not supported either.

Note in this case that ``cmake`` will not take configurations from your regular
``Makevars`` file (if you have such a file under ``~/.R/Makevars``) - instead, custom
configurations such as compilers to use and flags need to be set through CMake variables
like ``-DCMAKE_CXX_COMPILER``.


.. _r_gpu_support:

Building R package with GPU support
===================================

The procedure and requirements are similar as in :ref:`build_gpu_support`, so make sure to read it first.

On Linux, starting from the XGBoost directory type:

.. code-block:: bash

  cmake -B build -S . -DUSE_CUDA=ON -DR_LIB=ON
  cmake --build build --target install -j$(nproc)

When default target is used, an R package shared library would be built in the ``build`` area.
The ``install`` target, in addition, assembles the package files with this shared library under ``build/R-package`` and runs ``R CMD INSTALL``.

*********************
Building JVM Packages
*********************

Building XGBoost4J using Maven requires Maven 3 or newer, Java 7+ and CMake 3.18+ for
compiling Java code as well as the Java Native Interface (JNI) bindings. In addition, a
Python script is used during configuration, make sure the command ``python`` is available
on your system path (some distros use the name ``python3`` instead of ``python``).

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

  XGBoost4J-Spark now requires **Apache Spark 3.4+**. Latest versions of XGBoost4J-Spark uses facilities of `org.apache.spark.ml.param.shared` extensively to provide for a tight integration with Spark MLLIB framework, and these facilities are not fully available on earlier versions of Spark.

  Also, make sure to install Spark directly from `Apache website <https://spark.apache.org/>`_. **Upstream XGBoost is not guaranteed to work with third-party distributions of Spark, such as Cloudera Spark.** Consult appropriate third parties to obtain their distribution of XGBoost.

Additional System-dependent Features
====================================

- OpenMP on MacOS: See :ref:`running_cmake_and_build` for installing ``openmp``. The flag
  -``mvn -Duse.openmp=OFF`` can be used to disable OpenMP support.
- GPU support can be enabled by passing an additional flag to maven ``mvn -Duse.cuda=ON
  install``. See :ref:`build_gpu_support` for more info.

**************************
Building the Documentation
**************************

XGBoost uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ for documentation.  To
build it locally, you need a installed XGBoost with all its dependencies along with:

* System dependencies

  - git
  - graphviz

* Python dependencies

  Checkout the ``requirements.txt`` file under ``doc/``

Under ``xgboost/doc`` directory, run ``make <format>`` with ``<format>`` replaced by the
format you want.  For a list of supported formats, run ``make help`` under the same
directory.
