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

  * The binary wheel will support GPU algorithms (`gpu_hist`) on machines with NVIDIA GPUs. Please note that **training with multiple GPUs is only supported for Linux platform**. See :doc:`gpu/index`.
  * Currently, we provide binary wheels for 64-bit Linux, macOS and Windows.
  * Nightly builds are available. You can now run

    .. code-block:: bash

      pip install https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-[version]+[commithash]-py2.py3-none-manylinux1_x86_64.whl

    to install the nightly build with the given commit hash. See `this page
    <https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html>`_ to see the
    list of all nightly builds.

****************************
Building XGBoost from source
****************************
This page gives instructions on how to build and install XGBoost from scratch on various systems.

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

  - `Building on Linux Distributions`_
  - `Building on OSX`_
  - `Building on Windows`_
  - `Building with GPU support`_

* `Python Package Installation`_
* `R Package Installation`_
* `Trouble Shooting`_
* `Building the documentation`_

.. _build_shared_lib:

***************************
Building the Shared Library
***************************

Our goal is to build the shared library:

- On Linux/OSX the target library is ``libxgboost.so``
- On Windows the target library is ``xgboost.dll``

This shared library is used by different language bindings (with some additions depending
on the binding you choose).  For building language specific package, see corresponding
sections in this document.  The minimal building requirement is

- A recent C++ compiler supporting C++11 (g++-5.0 or higher)
- CMake 3.12 or higher.

For a list of CMake options, see ``#-- Options`` in CMakeLists.txt on top level of source tree.

Building on Linux distributions
===============================

On Ubuntu, one builds XGBoost by running CMake:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost
  cd xgboost
  mkdir build
  cd build
  cmake ..
  make -j$(nproc)

Building on OSX
===============

Install with pip: simple method
--------------------------------

First, obtain the OpenMP library (``libomp``) with Homebrew (https://brew.sh/) to enable multi-threading (i.e. using multiple CPU threads for training):

.. code-block:: bash

  brew install libomp

Then install XGBoost with ``pip``:

.. code-block:: bash

  pip3 install xgboost

You might need to run the command with ``--user`` flag if you run into permission errors.

Build from the source code - advanced method
--------------------------------------------

Obtain ``libomp`` from Homebrew:

.. code-block:: bash

  brew install libomp


Now clone the repository:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost

Create the ``build/`` directory and invoke CMake. After invoking CMake, you can build XGBoost with ``make``:

.. code-block:: bash

  mkdir build
  cd build
  cmake ..
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

Compile XGBoost with Microsoft Visual Studio
--------------------------------------------
To build with Visual Studio, we will need CMake. Make sure to install a recent version of CMake. Then run the following from the root of the XGBoost directory:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64"
  # for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
  # for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
  cmake --build . --config Release

This specifies an out of source build using the Visual Studio 64 bit generator. (Change the ``-G`` option appropriately if you have a different version of Visual Studio installed.)

After the build process successfully ends, you will find a ``xgboost.dll`` library file inside ``./lib/`` folder.

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

.. note:: Enabling distributed GPU training

  By default, distributed GPU training is disabled and only a single GPU will be used. To enable distributed GPU training, set the option ``USE_NCCL=ON``. Distributed GPU training depends on NCCL2, available at `this link <https://developer.nvidia.com/nccl>`_. Since NCCL2 is only available for Linux machines, **distributed GPU training is available only for Linux**.

  .. code-block:: bash

    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/path/to/nccl2
    make -j4

On Windows, run CMake as follows:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON

(Change the ``-G`` option appropriately if you have a different version of Visual Studio installed.)

.. note:: Visual Studio 2017 Win64 Generator may not work

  Choosing the Visual Studio 2017 generator may cause compilation failure. When it happens, specify the 2015 compiler by adding the ``-T`` option:

  .. code-block:: bash

    cmake .. -G"Visual Studio 15 2017 Win64" -T v140,cuda=8.0 -DUSE_CUDA=ON

To speed up compilation, the compute version specific to your GPU could be passed to cmake as, e.g., ``-DGPU_COMPUTE_VER=50``.
The above cmake configuration run will create an ``xgboost.sln`` solution file in the build directory. Build this solution in release mode as a x64 build, either from Visual studio or from command line:

.. code-block:: bash

  cmake --build . --target xgboost --config Release

To speed up compilation, run multiple jobs in parallel by appending option ``-- /MP``.

Makefiles
=========

It's only used for creating shorthands for running linters, performing packaging tasks
etc.  So the remaining makefiles are legacy.

Python Package Installation
===========================

The Python package is located at ``python-package/``.  There are several ways to build and
install the package from source:

1. Use Python setuptools directly

The XGBoost Python package supports most of the setuptools commands, here is a list of tested commands:

.. code-block:: bash

  python setup.py install  # Install the XGBoost to your current Python environment.
  python setup.py build    # Build the Python package.
  python setup.py build_ext # Build only the C++ core.
  python setup.py sdist     # Create a source distribution
  python setup.py bdist     # Create a binary distribution
  python setup.py bdist_wheel # Create a binary distribution with wheel format

Running ``python setup.py install`` will compile XGBoost using default CMake flags.  For
passing additional compilation options, append the flags to the command.  For example, to
enable CUDA acceleration and NCCL (distributed GPU) support:

.. code-block:: bash

  python setup.py install --use-cuda --use-nccl

Please refer to ``setup.py`` for a complete list of avaiable options.  Some other options
used for development are only available for using CMake directly.  See next section on
how to use CMake with setuptools manually.

You can install the created distribution packages using pip. For example, after running
``sdist`` setuptools command, a tar ball similar to ``xgboost-1.0.0.tar.gz`` will be
created under the ``dist`` directory.  Then you can install it by invoking the following
command under ``dist`` directory:

.. code-block:: bash

  # under python-package directory
  cd dist
  pip install ./xgboost-1.0.0.tar.gz


For details about these commands, please refer to the official document of `setuptools
<https://setuptools.readthedocs.io/en/latest/>`_, or just Google "how to install Python
package from source".  XGBoost Python package follows the general convention.  Setuptools
is usually available with your Python distribution, if not you can install it via system
command.  For example on Debian or Ubuntu:

.. code-block:: bash

  sudo apt-get install python-setuptools


For cleaning up the directory after running above commands, ``python setup.py clean`` is
not sufficient.  After copying out the build result, simply running ``git clean -xdf``
under ``python-package`` is an efficient way to remove generated cache files.  If you find
weird behaviors in Python build or running linter, it might be caused by those cached
files.

For using develop command (editable installation), see next section.

.. code-block::

  python setup.py develop   # Create a editable installation.
  pip install -e .          # Same as above, but carried out by pip.


2. Build C++ core with CMake first

This is mostly for C++ developers who don't want to go through the hooks in Python
setuptools.  You can build C++ library directly using CMake as described in above
sections.  After compilation, a shared object (or called dynamic linked library, jargon
depending on your platform) will appear in XGBoost's source tree under ``lib/`` directory.
On Linux distributions it's ``lib/libxgboost.so``.  From there all Python setuptools
commands will reuse that shared object instead of compiling it again.  This is especially
convenient if you are using the editable installation, where the installed package is
simply a link to the source tree.  We can perform rapid testing during development.  Here
is a simple bash script does that:

.. code-block:: bash

  # Under xgboost source tree.
  mkdir build
  cd build
  cmake ..
  make -j$(nproc)
  cd ../python-package
  pip install -e .  # or equivalently python setup.py develop

.. _mingw_python:

Building XGBoost library for Python for Windows with MinGW-w64 (Advanced)
-------------------------------------------------------------------------

Windows versions of Python are built with Microsoft Visual Studio. Usually Python binary modules are built with the same compiler the interpreter is built with. However, you may not be able to use Visual Studio, for following reasons:

1. VS is proprietary and commercial software. Microsoft provides a freeware "Community" edition, but its licensing terms impose restrictions as to where and how it can be used.
2. Visual Studio contains telemetry, as documented in `Microsoft Visual Studio Licensing Terms <https://visualstudio.microsoft.com/license-terms/mt736442/>`_. Running software with telemetry may be against the policy of your organization.

So you may want to build XGBoost with GCC own your own risk. This presents some difficulties because MSVC uses Microsoft runtime and MinGW-w64 uses own runtime, and the runtimes have different incompatible memory allocators. But in fact this setup is usable if you know how to deal with it. Here is some experience.

1. The Python interpreter will crash on exit if XGBoost was used. This is usually not a big issue.
2. ``-O3`` is OK.
3. ``-mtune=native`` is also OK.
4. Don't use ``-march=native`` gcc flag. Using it causes the Python interpreter to crash if the DLL was actually used.
5. You may need to provide the lib with the runtime libs. If ``mingw32/bin`` is not in ``PATH``, build a wheel (``python setup.py bdist_wheel``), open it with an archiver and put the needed dlls to the directory where ``xgboost.dll`` is situated. Then you can install the wheel with ``pip``.


R Package Installation
======================

Installing pre-packaged version
-------------------------------

You can install XGBoost from CRAN just like any other R package:

.. code-block:: R

   install.packages("xgboost")

.. note:: Using all CPU cores (threads) on Mac OSX

   If you are using Mac OSX, you should first install OpenMP library (``libomp``) by running

   .. code-block:: bash

      brew install libomp

   and then run ``install.packages("xgboost")``. Without OpenMP, XGBoost will only use a single CPU core, leading to suboptimal training speed.

Installing the development version
----------------------------------

Make sure you have installed git and a recent C++ compiler supporting C++11 (See above
sections for requirements of building C++ core).  On Windows, Rtools must be installed,
and its bin directory has to be added to ``PATH`` during the installation.

Due to the use of git-submodules, ``devtools::install_github`` can no longer be used to install the latest version of R package.
Thus, one has to run git to check out the code first:

.. code-block:: bash

  git clone --recursive https://github.com/dmlc/xgboost
  cd xgboost
  git submodule init
  git submodule update
  mkdir build
  cd build
  cmake .. -DR_LIB=ON
  make -j$(nproc)
  make install

If all fails, try `Building the shared library`_ to see whether a problem is specific to R
package or not.  Notice that the R package is installed by CMake directly.

.. _r_gpu_support:

Installing R package with GPU support
-------------------------------------

The procedure and requirements are similar as in `Building with GPU support`_, so make sure to read it first.

On Linux, starting from the XGBoost directory type:

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON -DR_LIB=ON
  make install -j$(nproc)

When default target is used, an R package shared library would be built in the ``build`` area.
The ``install`` target, in addition, assembles the package files with this shared library under ``build/R-package`` and runs ``R CMD INSTALL``.

On Windows, CMake with Visual C++ Build Tools (or Visual Studio) has to be used to build an R package with GPU support. Rtools must also be installed (perhaps, some other MinGW distributions with ``gendef.exe`` and ``dlltool.exe`` would work, but that was not tested).

.. code-block:: bash

  mkdir build
  cd build
  cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON -DR_LIB=ON
  cmake --build . --target install --config Release

When ``--target xgboost`` is used, an R package DLL would be built under ``build/Release``.
The ``--target install``, in addition, assembles the package files with this dll under ``build/R-package`` and runs ``R CMD INSTALL``.

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


Building the Documentation
==========================
XGBoost uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ for documentation.  To build it locally, you need a installed XGBoost with all its dependencies along with:

* System dependencies

  - git
  - graphviz

* Python dependencies

  - sphinx
  - breathe
  - guzzle_sphinx_theme
  - recommonmark
  - mock
  - sh
  - graphviz
  - matplotlib

Under ``xgboost/doc`` directory, run ``make <format>`` with ``<format>`` replaced by the format you want.  For a list of supported formats, run ``make help`` under the same directory.
