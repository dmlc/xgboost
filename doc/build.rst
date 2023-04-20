####################
Building From Source
####################

This page gives instructions on how to build and install XGBoost from the source code on various
systems.  If the instructions do not work for you, please feel free to ask questions at
`the user forum <https://discuss.xgboost.ai>`_.


.. note:: Pre-built binary is available: now with GPU support

  Consider installing XGBoost from a pre-built binary, to avoid the trouble of building XGBoost from the source.  Checkout :doc:`Installation Guide </install>`.

.. contents:: Contents
  :local:

.. _get_source:

*************************
Obtaining the Source Code
*************************
To obtain the development repository of XGBoost, one needs to use ``git``.

.. note:: Use of Git submodules

  XGBoost uses Git submodules to manage dependencies. So when you clone the repo, remember to specify ``--recursive`` option:

  .. code-block:: bash

    git clone --recursive https://github.com/dmlc/xgboost

For windows users who use github tools, you can open the git shell and type the following command:

.. code-block:: batch

  git submodule init
  git submodule update


.. _build_shared_lib:

***************************
Building the Shared Library
***************************

This section describes the procedure to build the shared library and CLI interface
independently.  For building language specific package, see corresponding sections in this
document.

- On Linux and other UNIX-like systems, the target library is ``libxgboost.so``
- On MacOS, the target library is ``libxgboost.dylib``
- On Windows the target library is ``xgboost.dll``

This shared library is used by different language bindings (with some additions depending
on the binding you choose).  The minimal building requirement is

- A recent C++ compiler supporting C++11 (g++-5.0 or higher)
- CMake 3.14 or higher.

For a list of CMake options like GPU support, see ``#-- Options`` in CMakeLists.txt on top
level of source tree.

Building on Linux and other UNIX-like systems
=============================================

After obtaining the source code, one builds XGBoost by running CMake:

.. code-block:: bash

  cd xgboost
  mkdir build
  cd build
  cmake ..
  make -j$(nproc)

Building on MacOS
=================

Obtain ``libomp`` from `Homebrew <https://brew.sh/>`_:

.. code-block:: bash

  brew install libomp

Rest is the same as building on Linux.


Building on Windows
===================

XGBoost support compilation with Microsoft Visual Studio and MinGW.  To build with Visual
Studio, we will need CMake. Make sure to install a recent version of CMake. Then run the
following from the root of the XGBoost directory:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64"
  # for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
  # for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
  cmake --build . --config Release

This specifies an out of source build using the Visual Studio 64 bit generator. (Change the ``-G`` option appropriately if you have a different version of Visual Studio installed.)

After the build process successfully ends, you will find a ``xgboost.dll`` library file
inside ``./lib/`` folder.  Some notes on using MinGW is added in :ref:`python_mingw`.

.. _build_gpu_support:


Building with GPU support
=========================

XGBoost can be built with GPU support for both Linux and Windows using CMake. See
`Building R package with GPU support`_ for special instructions for R.

An up-to-date version of the CUDA toolkit is required.

.. note:: Checking your compiler version

  CUDA is really picky about supported compilers, a table for the compatible compilers for the latests CUDA version on Linux can be seen `here <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

  Some distros package a compatible ``gcc`` version with CUDA. If you run into compiler errors with ``nvcc``, try specifying the correct compiler with ``-DCMAKE_CXX_COMPILER=/path/to/correct/g++ -DCMAKE_C_COMPILER=/path/to/correct/gcc``. On Arch Linux, for example, both binaries can be found under ``/opt/cuda/bin/``.

From the command line on Linux starting from the XGBoost directory:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON
  make -j4

.. note:: Specifying compute capability

  To speed up compilation, the compute version specific to your GPU could be passed to cmake as, e.g., ``-DGPU_COMPUTE_VER=50``. A quick explanation and numbers for some architectures can be found `in this page <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.

.. note:: Faster distributed GPU training with NCCL

  By default, distributed GPU training is enabled and uses Rabit for communication. For faster training, set the option ``USE_NCCL=ON``. Faster distributed GPU training depends on NCCL2, available at `this link <https://developer.nvidia.com/nccl>`_. Since NCCL2 is only available for Linux machines, **faster distributed GPU training is available only for Linux**.

  .. code-block:: bash

    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/path/to/nccl2
    make -j4

On Windows, run CMake as follows:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON

(Change the ``-G`` option appropriately if you have a different version of Visual Studio installed.)

The above cmake configuration run will create an ``xgboost.sln`` solution file in the build directory. Build this solution in Release mode, either from Visual studio or from command line:

.. code-block:: bash

  cmake --build . --target xgboost --config Release

To speed up compilation, run multiple jobs in parallel by appending option ``-- /MP``.

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

  You can navigate to ``python-package/`` directory and install the Python package directly
  by running

  .. code-block:: console

    $ cd python-package/
    $ pip install -v .

  which will compile XGBoost's native (C++) code using default CMake flags.
  To enable additional compilation options, pass corresponding ``--config-settings``:

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

  To further enable rapid development and iteration, we provide an **editable installation**.
  In an editable installation, the installed package is simply a symbolic link to your
  working copy of the XGBoost source code. So every changes you make to your source
  directory will be immediately visible to the Python interpreter. Here is how to
  install XGBoost as editable installation:

  .. code-block:: bash

    # Under xgboost source directory
    mkdir build
    cd build
    # Build shared library libxgboost.so
    cmake .. -GNinja
    ninja
    # Install as editable installation
    cd ../python-package
    pip install -e .

4. Use ``libxgboost.so`` on system path.

  This option is useful for package managers that wish to separately package
  ``libxgboost.so`` and the XGBoost Python package. For example, Conda
  publishes ``libxgboost`` (for the shared library) and ``py-xgboost``
  (for the Python package).

  To use this option, first make sure that ``libxgboost.so`` exists in the system library path:

  .. code-block:: python

    import sys
    import pathlib
    libpath = pathlib.Path(sys.prefix).joinpath("lib", "libxgboost.so")
    assert libpath.exists()

  Then pass ``use_system_libxgboost=True`` option to ``pip install``:

  .. code-block:: bash

    cd python-package
    pip install . --config-settings use_system_libxgboost=True


.. note::

  See :doc:`contrib/python_packaging` for instructions on packaging
  and distributing XGBoost as Python distributions.

.. _python_mingw:

Building Python Package for Windows with MinGW-w64 (Advanced)
=============================================================

Windows versions of Python are built with Microsoft Visual Studio. Usually Python binary modules are built with the same compiler the interpreter is built with. However, you may not be able to use Visual Studio, for following reasons:

1. VS is proprietary and commercial software. Microsoft provides a freeware "Community" edition, but its licensing terms impose restrictions as to where and how it can be used.
2. Visual Studio contains telemetry, as documented in `Microsoft Visual Studio Licensing Terms <https://visualstudio.microsoft.com/license-terms/mt736442/>`_. Running software with telemetry may be against the policy of your organization.

So you may want to build XGBoost with GCC own your own risk. This presents some difficulties because MSVC uses Microsoft runtime and MinGW-w64 uses own runtime, and the runtimes have different incompatible memory allocators. But in fact this setup is usable if you know how to deal with it. Here is some experience.

1. The Python interpreter will crash on exit if XGBoost was used. This is usually not a big issue.
2. ``-O3`` is OK.
3. ``-mtune=native`` is also OK.
4. Don't use ``-march=native`` gcc flag. Using it causes the Python interpreter to crash if the DLL was actually used.
5. You may need to provide the lib with the runtime libs. If ``mingw32/bin`` is not in ``PATH``, build a wheel (``pip wheel``), open it with an archiver and put the needed dlls to the directory where ``xgboost.dll`` is situated. Then you can install the wheel with ``pip``.

******************************
Building R Package From Source
******************************

By default, the package installed by running ``install.packages`` is built from source.
Here we list some other options for installing development version.

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

But if you want to use CMake build for better performance (which has the logic for
detecting available CPU instructions) or greater flexibility around compile flags, the
above snippet can be replaced by:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DR_LIB=ON
  make -j$(nproc)
  make install


Installing the development version with Visual Studio (Windows)
===============================================================

On Windows, CMake with Visual C++ Build Tools (or Visual Studio) can be used to build the R package.

While not required, this build can be faster if you install the R package ``processx`` with ``install.packages("processx")``.

.. note:: Setting correct PATH environment variable on Windows

  If you are using Windows, make sure to include the right directories in the PATH environment variable.

  * If you are using R 4.x with RTools 4.0:
    - ``C:\rtools40\usr\bin``
    - ``C:\rtools40\mingw64\bin``

  * If you are using R 3.x with RTools 3.x:

    - ``C:\Rtools\bin``
    - ``C:\Rtools\mingw_64\bin``

Open the Command Prompt and navigate to the XGBoost directory, and then run the following commands. Make sure to specify the correct R version.

.. code-block:: bash

  cd C:\path\to\xgboost
  mkdir build
  cd build
  cmake .. -G"Visual Studio 16 2019" -A x64 -DR_LIB=ON -DR_VERSION=4.0.0
  cmake --build . --target install --config Release


.. _r_gpu_support:

Building R package with GPU support
===================================

The procedure and requirements are similar as in :ref:`build_gpu_support`, so make sure to read it first.

On Linux, starting from the XGBoost directory type:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON -DR_LIB=ON
  make install -j$(nproc)

When default target is used, an R package shared library would be built in the ``build`` area.
The ``install`` target, in addition, assembles the package files with this shared library under ``build/R-package`` and runs ``R CMD INSTALL``.

On Windows, CMake with Visual Studio has to be used to build an R package with GPU support. Rtools must also be installed.

.. note:: Setting correct PATH environment variable on Windows

  If you are using Windows, make sure to include the right directories in the PATH environment variable.

  * If you are using R 4.x with RTools 4.0:

    - ``C:\rtools40\usr\bin``
    - ``C:\rtools40\mingw64\bin``
  * If you are using R 3.x with RTools 3.x:

    - ``C:\Rtools\bin``
    - ``C:\Rtools\mingw_64\bin``

Open the Command Prompt and navigate to the XGBoost directory, and then run the following commands. Make sure to specify the correct R version.

.. code-block:: bash

  cd C:\path\to\xgboost
  mkdir build
  cd build
  cmake .. -G"Visual Studio 16 2019" -A x64 -DUSE_CUDA=ON -DR_LIB=ON -DR_VERSION=4.0.0
  cmake --build . --target install --config Release

If CMake can't find your R during the configuration step, you might provide the location of R to CMake like this: ``-DLIBR_HOME="C:\Program Files\R\R-4.0.0"``.

If on Windows you get a "permission denied" error when trying to write to ...Program Files/R/... during the package installation, create a ``.Rprofile`` file in your personal home directory (if you don't already have one in there), and add a line to it which specifies the location of your R packages user library, like the following:

.. code-block:: R

  .libPaths( unique(c("C:/Users/USERNAME/Documents/R/win-library/3.4", .libPaths())))

You might find the exact location by running ``.libPaths()`` in R GUI or RStudio.


*********************
Building JVM Packages
*********************

Building XGBoost4J using Maven requires Maven 3 or newer, Java 7+ and CMake 3.13+ for compiling Java code as well as the Java Native Interface (JNI) bindings.

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
==========================
If you are on Mac OS and using a compiler that supports OpenMP, you need to go to the file ``xgboost/jvm-packages/create_jni.py`` and comment out the line

.. code-block:: python

  CONFIG["USE_OPENMP"] = "OFF"

in order to get the benefit of multi-threading.

Building with GPU support
==========================
If you want to build XGBoost4J that supports distributed GPU training, run

.. code-block:: bash

  mvn -Duse.cuda=ON install

**************************
Building the Documentation
**************************
XGBoost uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ for documentation.  To build it locally, you need a installed XGBoost with all its dependencies along with:

* System dependencies

  - git
  - graphviz

* Python dependencies

  Checkout the ``requirements.txt`` file under ``doc/``

Under ``xgboost/doc`` directory, run ``make <format>`` with ``<format>`` replaced by the format you want.  For a list of supported formats, run ``make help`` under the same directory.
