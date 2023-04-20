###########################################
Notes on packaging XGBoost's Python package
###########################################


.. contents:: Contents
  :local:

.. _packaging_python_xgboost:

***************************************************
How to build binary wheels and source distributions
***************************************************

Wheels and source distributions (sdist for short) are the two main
mechanisms for packaging and distributing Python packages.

* A **source distribution** (sdist) is a tarball (``.tar.gz`` extension) that
  contains the source code.
* A **wheel** is a ZIP-compressed archive (with ``.whl`` extension)
  representing a *built* distribution. Unlike an sdist, a wheel can contain
  compiled components. The compiled components are compiled prior to distribution,
  making it more convenient for end-users to install a wheel. Wheels containing
  compiled components are referred to as **binary wheels**.

See `Python Packaging User Guide <https://packaging.python.org/en/latest/>`_
to learn more about how Python packages in general are packaged and
distributed.

For the remainder of this document, we will focus on packaging and
distributing XGBoost.

Building sdists
===============

In the case of XGBoost, an sdist contains both the Python code as well as
the C++ code, so that the core part of XGBoost can be compiled into the
shared libary ``libxgboost.so`` [#shared_lib_name]_.

You can obtain an sdist as follows:

.. code-block:: console

  $ python -m build --sdist .

(You'll need to install the ``build`` package first:
``pip install build`` or ``conda install python-build``.)

Running ``pip install`` with an sdist will launch CMake and a C++ compiler
to compile the bundled C++ code into ``libxgboost.so``:

.. code-block:: console

  $ pip install -v xgboost-2.0.0.tar.gz  # Add -v to show build progress

Building binary wheels
======================

You can also build a wheel as follows:

.. code-block:: console

   $ pip wheel --no-deps -v .

Notably, the resulting wheel contains a copy of the shared library
``libxgboost.so`` [#shared_lib_name]_. The wheel is a **binary wheel**,
since it contains a compiled binary.


Running ``pip install`` with the binary wheel will extract the content of
the wheel into the current Python environment. Since the wheel already
contains a pre-built copy of ``libxgboost.so``, it does not have to be
built at the time of install. So ``pip install`` with the binary wheel
completes quickly:

.. code-block:: console
  
  $ pip install xgboost-2.0.0-py3-none-linux_x86_64.whl  # Completes quickly

.. rubric:: Footnotes

.. [#shared_lib_name] The name of the shared library file will differ
   depending on the operating system in use. See :ref:`build_shared_lib`.
