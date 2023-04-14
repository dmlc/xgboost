#########################
Notes on Python packaging
#########################


.. contents:: Contents
  :local:

.. _binary_wheels:

***************************************************
How to build binary wheels and source distributions
***************************************************

Binary wheels and source distributions (sdist for short) are two main
mechanisms for packaging and distributing Python packages.

A **source distribution** (sdist) is a tarball (``.tar.gz`` extension) that
contains source code. In the case of XGBoost, an sdist contains
both the Python code as well as the C++. You can obtain an sdist as follows:

.. code-block:: console

  $ python -m build --sdist .

(You'll need to install the ``build`` package first:
``pip install build`` or ``conda install python-build``.)

Running ``pip install`` with an sdist will launch CMake and a C++ compiler
to compile the bundled C++ code into native library ``libxgboost.so``:

.. code-block:: console

  $ pip install -v xgboost-2.0.0.tar.gz  # Add -v to show build progress

A **binary wheel** is a ZIP-compressed archive (``.whl`` extension) that
contains Python source code as well as compiled extensions. In the case of
XGBoost, a binary wheel contains the Python code as well as a pre-built
native library ``libxgboost.so``. Build a binary wheel as follows:

.. code-block:: console

   $ pip wheel --no-deps -v .

Running ``pip install`` with a binary wheel will extract the content of
the wheel into the current Python environment. Crucially, since the
wheel already contains a pre-built native library ``libxgboost.so``,
it does not have to be built. So ``pip install`` with a binary wheel
completes quickly.

.. code-block:: console
  
  $ pip install -v xgboost-2.0.0-py3-none-linux_x86_64.whl

.. note:: Bundling OpenMP library on Windows

  XGBoost uses OpenMP to implement parallel algorithms on CPUs.
  Consequently, on the Windows platform, XGBoost requires access
  to the system library ``vcomp140.dll``. Not every Windows
  machine has this library installed. So we have two choices
  when it comes to distributing XGBoost:

  1. Ask all users to install
     `Visual C++ Redistributable for Visual Studio 2015
     <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_.
  2. Inject ``vcomp140.dll`` into the binary wheel. In this
     case, ``vcomp140.dll`` will be installed in the same directory
     as XGBoost. To enable bundling, pass ``bundle_vcomp140_dll``
     option to Pip:
     
     .. code-block:: console

       $ # Use Pip 22.1+
       $ pip install . --config-settings bundle_vcomp140_dll=True

  The XGBoost project uses Option 2: we bundle ``vcomp140.dll``
  in the binary wheel targeting Windows, before we upload it to
  Python Packaging Index (PyPI).
