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
