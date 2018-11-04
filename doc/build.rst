##################
Installation Guide
##################

.. note:: Pre-built binary wheel for Python

  If you are planning to use Python, consider installing XGBoost from a pre-built binary wheel, available from Python Package Index (PyPI). You may download and install it by running

  .. code-block:: bash

    # Ensure that you are downloading one of the following:
    #   * xgboost-{version}-py2.py3-none-manylinux1_x86_64.whl
    #   * xgboost-{version}-py2.py3-none-win_amd64.whl
    pip3 install xgboost

  * The binary wheel will support GPU algorithms (`gpu_exact`, `gpu_hist`) on machines with NVIDIA GPUs. **However, it will not support multi-GPU training; only single GPU will be used.** To enable multi-GPU training, download and install the binary wheel from `this page <https://s3-us-west-2.amazonaws.com/xgboost-wheels/list.html>`_.
  * Currently, we provide binary wheels for 64-bit Linux and Windows.

****************************
Building XGBoost from source
****************************
This page gives instructions on how to build and install XGBoost from scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (``libxgboost.so`` for Linux/OSX and ``xgboost.dll`` for Windows).
   (For R-package installation, please directly refer to `R Package Installation`_.)
2. Then install the language packages (e.g. Python Package).

.. note:: Use of Git submodules

  XGBoost uses Git submodules to manage dependencies. So when you clone the repo, remember to specify ``--recursive`` option:
  
  .. code-block:: bash

    git clone --recursive https://github.com/dmlc/xgboost

For windows users who use github tools, you can open the git shell and type the following command:

.. code-block:: batch

  git submodule init
  git submodule update

Please refer to `Trouble Shooting`_ section first if you have any problem
during installation. If the instructions do not work for you, please feel free
to ask questions at `the user forum <https://discuss.xgboost.ai>`_.

**Contents**

* `Building the Shared Library`_

  - `Building on Ubuntu/Debian`_
  - `Building on OSX`_
  - `Building on Windows`_
  - `Building with GPU support`_
  - `Customized Building`_

* `Python Package Installation`_
* `R Package Installation`_
* `Trouble Shooting`_

***************************
Building the Shared Library
***************************

Our goal is to build the shared library:

- On Linux/OSX the target library is ``libxgboost.so``
- On Windows the target library is ``xgboost.dll``

The minimal building requirement is

- A recent C++ compiler supporting C++11 (g++-4.8 or higher)

We can edit ``make/config.mk`` to change the compile options, and then build by
``make``. If everything goes well, we can go to the specific language installation section.

Building on Ubuntu/Debian
=========================

On Ubuntu, one builds XGBoost by running

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost
  cd xgboost; make -j4

Building on OSX
===============

Install with pip: simple method
--------------------------------

First, obtain ``gcc-7`` with Homebrew (https://brew.sh/) to enable multi-threading (i.e. using multiple CPU threads for training). The default Apple Clang compiler does not support OpenMP, so using the default compiler would have disabled multi-threading.

.. code-block:: bash

  brew install gcc@7

Then install XGBoost with ``pip``:

.. code-block:: bash

  pip3 install xgboost

You might need to run the command with ``--user`` flag if you run into permission errors.

Build from the source code - advanced method
--------------------------------------------

Obtain ``gcc-7`` from Homebrew:

.. code-block:: bash

  brew install gcc@7

Now clone the repository:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost

Create the ``build/`` directory and invoke CMake. Make sure to add ``CC=gcc-7 CXX=g++-7`` so that Homebrew GCC is selected. After invoking CMake, you can build XGBoost with ``make``:

.. code-block:: bash

  mkdir build
  cd build
  CC=gcc-7 CXX=g++-7 cmake ..
  make -j4

You may now continue to `Python Package Installation`_.

Building on Windows
===================
You need to first clone the XGBoost repo with ``--recursive`` option, to clone the submodules.
We recommend you use `Git for Windows <https://git-for-windows.github.io/>`_, as it comes with a standard Bash shell. This will highly ease the installation process.

.. code-block:: bash

  git submodule init
  git submodule update

XGBoost support compilation with Microsoft Visual Studio and MinGW.

Compile XGBoost using MinGW
---------------------------
After installing `Git for Windows <https://git-for-windows.github.io/>`_, you should have a shortcut named ``Git Bash``. You should run all subsequent steps in ``Git Bash``.

In MinGW, ``make`` command comes with the name ``mingw32-make``. You can add the following line into the ``.bashrc`` file:

.. code-block:: bash

  alias make='mingw32-make'

(On 64-bit Windows, you should get `MinGW64 <https://sourceforge.net/projects/mingw-w64/>`_ instead.) Make sure
that the path to MinGW is in the system PATH.

To build with MinGW, type:

.. code-block:: bash

  cp make/mingw64.mk config.mk; make -j4

See :ref:`mingw_python` for buildilng XGBoost for Python.

Compile XGBoost with Microsoft Visual Studio
--------------------------------------------
To build with Visual Studio, we will need CMake. Make sure to install a recent version of CMake. Then run the following from the root of the XGBoost directory:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 12 2013 Win64"

This specifies an out of source build using the MSVC 12 64 bit generator. Open the ``.sln`` file in the build directory and build with Visual Studio. To use the Python module you can copy ``xgboost.dll`` into ``python-package/xgboost``.

After the build process successfully ends, you will find a ``xgboost.dll`` library file inside ``./lib/`` folder, copy this file to the the API package folder like ``python-package/xgboost`` if you are using Python API.

Unofficial windows binaries and instructions on how to use them are hosted on `Guido Tapia's blog <http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/>`_.

.. _build_gpu_support:

Building with GPU support
=========================
XGBoost can be built with GPU support for both Linux and Windows using CMake. GPU support works with the Python package as well as the CLI version. See `Installing R package with GPU support`_ for special instructions for R.

An up-to-date version of the CUDA toolkit is required.

From the command line on Linux starting from the XGBoost directory:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON
  make -j4

.. note:: Enabling multi-GPU training

  By default, multi-GPU training is disabled and only a single GPU will be used. To enable multi-GPU training, set the option ``USE_NCCL=ON``. Multi-GPU training depends on NCCL2, available at `this link <https://developer.nvidia.com/nccl>`_. Since NCCL2 is only available for Linux machines, **multi-GPU training is available only for Linux**.

  .. code-block:: bash

    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/path/to/nccl2
    make -j4

On Windows, see what options for generators you have for CMake, and choose one with ``[arch]`` replaced with Win64:

.. code-block:: bash

  cmake -help

Then run CMake as follows:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON

.. note:: Visual Studio 2017 Win64 Generator may not work

  Choosing the Visual Studio 2017 generator may cause compilation failure. When it happens, specify the 2015 compiler by adding the ``-T`` option:

  .. code-block:: bash

    make .. -G"Visual Studio 15 2017 Win64" -T v140,cuda=8.0 -DR_LIB=ON -DUSE_CUDA=ON

To speed up compilation, the compute version specific to your GPU could be passed to cmake as, e.g., ``-DGPU_COMPUTE_VER=50``.
The above cmake configuration run will create an ``xgboost.sln`` solution file in the build directory. Build this solution in release mode as a x64 build, either from Visual studio or from command line:

.. code-block:: bash

  cmake --build . --target xgboost --config Release

To speed up compilation, run multiple jobs in parallel by appending option ``-- /MP``.

Customized Building
===================

The configuration file ``config.mk`` modifies several compilation flags:
- Whether to enable support for various distributed filesystems such as HDFS and Amazon S3
- Which compiler to use
- And some more

To customize, first copy ``make/config.mk`` to the project root and then modify the copy.

Alternatively, use CMake.

Python Package Installation
===========================

The Python package is located at ``python-package/``.
There are several ways to install the package:

1. Install system-wide, which requires root permission:

.. code-block:: bash

  cd python-package; sudo python setup.py install

You will however need Python ``distutils`` module for this to
work. It is often part of the core Python package or it can be installed using your
package manager, e.g. in Debian use

.. code-block:: bash

  sudo apt-get install python-setuptools

.. note:: Re-compiling XGBoost

  If you recompiled XGBoost, then you need to reinstall it again to make the new library take effect.

2. Only set the environment variable ``PYTHONPATH`` to tell Python where to find
   the library. For example, assume we cloned `xgboost` on the home directory
   `~`. then we can added the following line in `~/.bashrc`.
   This option is **recommended for developers** who change the code frequently. The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ``setup`` again)

.. code-block:: bash

  export PYTHONPATH=~/xgboost/python-package

3. Install only for the current user.

.. code-block:: bash

  cd python-package; python setup.py develop --user

4. If you are installing the latest XGBoost version which requires compilation, add MinGW to the system PATH:

.. code-block:: bash

    import os
    os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

.. _mingw_python:

Building XGBoost library for Python for Windows with MinGW-w64
--------------------------------------------------------------

Windows versions of Python are built with Microsoft Visual Studio. Usually Python binary modules are built with the same compiler the interpreter is built with, raising several potential concerns.

1. VS is proprietary and commercial software. Microsoft provides a freeware "Community" edition, but its licensing terms are unsuitable for many organizations.
2. Visual Studio contains telemetry, as documented in `Microsoft Visual Studio Licensing Terms <https://visualstudio.microsoft.com/license-terms/mt736442/>`_. It `has been inserting telemetry <https://old.reddit.com/r/cpp/comments/4ibauu/visual_studio_adding_telemetry_function_calls_to/>`_ into apps for some time. In order to download VS distribution from MS servers one has to run the application containing telemetry. These facts have raised privacy and security concerns among some users and system administrators. Running software with telemetry may be against the policy of your organization.
3. g++ usually generates faster code on ``-O3``.

So you may want to build XGBoost with g++ own your own risk. This opens a can of worms, because MSVC uses Microsoft runtime and MinGW-w64 uses own runtime, and the runtimes have different incompatible memory allocators. But in fact this setup is usable if you know how to deal with it. Here is some experience.

1. The Python interpreter will crash on exit if XGBoost was used. This is usually not a big issue.
2. ``-O3`` is OK.
3. ``-mtune=native`` is also OK.
4. Don't use ``-march=native`` gcc flag. Using it causes the Python interpreter to crash if the dll was actually used.
5. You may need to provide the lib with the runtime libs. If ``mingw32/bin`` is not in ``PATH``, build a wheel (``python setup.py bdist_wheel``), open it with an archiver and put the needed dlls to the directory where ``xgboost.dll`` is situated. Then you can install the wheel with ``pip``.

R Package Installation
======================

Installing pre-packaged version
-------------------------------

You can install xgboost from CRAN just like any other R package:

.. code-block:: R

  install.packages("xgboost")

For OSX users, single-threaded version will be installed. So only one thread will be used for training. To enable use of multiple threads (and utilize capacity of multi-core CPUs), see the section :ref:`osx_multithread` to install XGBoost from source.

Installing the development version
----------------------------------

Make sure you have installed git and a recent C++ compiler supporting C++11 (e.g., g++-4.8 or higher).
On Windows, Rtools must be installed, and its bin directory has to be added to ``PATH`` during the installation.

Due to the use of git-submodules, ``devtools::install_github`` can no longer be used to install the latest version of R package.
Thus, one has to run git to check out the code first:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost
  cd xgboost
  git submodule init
  git submodule update
  cd R-package
  R CMD INSTALL .

If the last line fails because of the error ``R: command not found``, it means that R was not set up to run from command line.
In this case, just start R as you would normally do and run the following:

.. code-block:: R

  setwd('wherever/you/cloned/it/xgboost/R-package/')
  install.packages('.', repos = NULL, type="source")

The package could also be built and installed with cmake (and Visual C++ 2015 on Windows) using instructions from the next section, but without GPU support (omit the ``-DUSE_CUDA=ON`` cmake parameter).

If all fails, try `Building the shared library`_ to see whether a problem is specific to R package or not.

.. _osx_multithread:

Installing R package on Mac OSX with multi-threading
----------------------------------------------------

First, obtain ``gcc-7`` with Homebrew (https://brew.sh/) to enable multi-threading (i.e. using multiple CPU threads for training). The default Apple Clang compiler does not support OpenMP, so using the default compiler would have disabled multi-threading.

.. code-block:: bash

  brew install gcc@7

Now, clone the repository:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost

Create the ``build/`` directory and invoke CMake with option ``R_LIB=ON``. Make sure to add ``CC=gcc-7 CXX=g++-7`` so that Homebrew GCC is selected. After invoking CMake, you can install the R package by running ``make`` and ``make install``:

.. code-block:: bash

  mkdir build
  cd build
  CC=gcc-7 CXX=g++-7 cmake .. -DR_LIB=ON
  make -j4
  make install

Installing R package with GPU support
-------------------------------------

The procedure and requirements are similar as in `Building with GPU support`_, so make sure to read it first.

On Linux, starting from the XGBoost directory type:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON -DR_LIB=ON
  make install -j

When default target is used, an R package shared library would be built in the ``build`` area.
The ``install`` target, in addition, assembles the package files with this shared library under ``build/R-package``, and runs ``R CMD INSTALL``.

On Windows, CMake with Visual C++ Build Tools (or Visual Studio) has to be used to build an R package with GPU support. Rtools must also be installed (perhaps, some other MinGW distributions with ``gendef.exe`` and ``dlltool.exe`` would work, but that was not tested).

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON -DR_LIB=ON
  cmake --build . --target install --config Release

When ``--target xgboost`` is used, an R package dll would be built under ``build/Release``.
The ``--target install``, in addition, assembles the package files with this dll under ``build/R-package``, and runs ``R CMD INSTALL``.

If cmake can't find your R during the configuration step, you might provide the location of its executable to cmake like this: ``-DLIBR_EXECUTABLE="C:/Program Files/R/R-3.4.1/bin/x64/R.exe"``.

If on Windows you get a "permission denied" error when trying to write to ...Program Files/R/... during the package installation, create a ``.Rprofile`` file in your personal home directory (if you don't already have one in there), and add a line to it which specifies the location of your R packages user library, like the following:

.. code-block:: R

  .libPaths( unique(c("C:/Users/USERNAME/Documents/R/win-library/3.4", .libPaths())))

You might find the exact location by running ``.libPaths()`` in R GUI or RStudio.

Trouble Shooting
================

1. Compile failed after ``git pull``

   Please first update the submodules, clean all and recompile:

   .. code-block:: bash

     git submodule update && make clean_all && make -j4

2. Compile failed after ``config.mk`` is modified

   Need to clean all first:

   .. code-block:: bash

     make clean_all && make -j4

3. ``Makefile: dmlc-core/make/dmlc.mk: No such file or directory``

   We need to recursively clone the submodule:

   .. code-block:: bash

     git submodule init
     git submodule update

   Alternatively, do another clone

   .. code-block:: bash

     git clone https://github.com/dmlc/xgboost --recursive

