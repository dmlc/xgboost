########################
Adding and running tests
########################

A high-quality suite of tests is crucial in ensuring correctness and robustness of the codebase. Here, we provide instructions how to run unit tests, and also how to add a new one.

**Contents**

.. contents::
  :backlinks: none
  :local:

**********************
Adding a new unit test
**********************

Python package: pytest
======================
Add your test under the directories

- `tests/python/ <https://github.com/dmlc/xgboost/tree/master/tests/python>`_
- `tests/python-gpu/ <https://github.com/dmlc/xgboost/tree/master/tests/python-gpu>`_ (if you are testing GPU code)
- `tests/test_distributed <https://github.com/dmlc/xgboost/tree/master/tests/test_distributed>`_. (if a distributed framework is used)

Refer to `the PyTest tutorial <https://docs.pytest.org/en/latest/getting-started.html>`_
to learn how to write tests for Python code.

You may try running your test by following instructions in :ref:`this section <running_pytest>`.

C++: Google Test
================
Add your test under the directory `tests/cpp/ <https://github.com/dmlc/xgboost/tree/master/tests/cpp>`_. Refer to `this excellent tutorial on using Google Test <https://developer.ibm.com/articles/au-googletestingframework/>`_.

You may try running your test by following instructions in :ref:`this section <running_gtest>`. Note. Google Test version 1.8.1 or later is required.

JVM packages: JUnit / scalatest
===============================
The JVM packages for XGBoost (XGBoost4J / XGBoost4J-Spark) use `the Maven Standard Directory Layout <https://maven.apache.org/guides/introduction/introduction-to-the-standard-directory-layout.html>`_. Specifically, the tests for the JVM packages are located in the following locations:

* `jvm-packages/xgboost4j/src/test/ <https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j/src/test>`_
* `jvm-packages/xgboost4j-spark/src/test/ <https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-spark/src/test>`_

To write a test for Java code, see `JUnit 5 tutorial <https://junit.org/junit5/docs/current/user-guide/>`_.
To write a test for Scala, see `Scalatest tutorial <http://www.scalatest.org/user_guide/writing_your_first_test>`_.

You may try running your test by following instructions in :ref:`this section <running_jvm_tests>`.

R package: testthat
===================
Add your test under the directory `R-package/tests/testthat <https://github.com/dmlc/xgboost/tree/master/R-package/tests/testthat>`_. Refer to `this excellent tutorial on testthat <https://kbroman.org/pkg_primer/pages/tests.html>`_.

You may try running your test by following instructions in :ref:`this section <running_r_tests>`.

**************************
Running Unit Tests Locally
**************************

.. _running_r_tests:

R package
=========
Run

.. code-block:: bash

  python ./ops/script/test_r_package.py --task=check

at the root of the project directory. The command builds and checks the XGBoost
r-package. Alternatively, if you want to just run the tests, you can use the following
commands after installing XGBoost:

.. code-block:: bash

  cd R-package/tests/
  Rscript testthat.R

.. _running_jvm_tests:

JVM packages
============
Maven is used

.. code-block:: bash

  mvn test

.. _running_pytest:

Python package: pytest
======================

To run Python unit tests, first install `pytest <https://docs.pytest.org/en/latest/contents.html>`_ package:

.. code:: bash

  pip3 install pytest

Then compile XGBoost according to instructions in :ref:`build_shared_lib`. Finally, invoke pytest at the project root directory:

.. code:: bash

  # Tell Python where to find XGBoost module
  export PYTHONPATH=./python-package
  pytest -v -s --fulltrace tests/python

In addition, to test CUDA code, run:

.. code:: bash

  # Tell Python where to find XGBoost module
  export PYTHONPATH=./python-package
  pytest -v -s --fulltrace tests/python-gpu

(For this step, you should have compiled XGBoost with CUDA enabled.)

For testing with distributed frameworks like ``Dask`` and ``PySpark``:

.. code:: bash

  # Tell Python where to find XGBoost module
  export PYTHONPATH=./python-package
  pytest -v -s --fulltrace tests/test_distributed

.. _running_gtest:

C++: Google Test
================

To build and run C++ unit tests enable tests while running CMake:

.. code-block:: bash

  cmake -B build -S . -GNinja -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON
  cmake --build build
  cd ./build
  ./testxgboost

Flags like ``USE_CUDA``, ``USE_DMLC_GTEST`` are optional. For more info about how to build
XGBoost from source, see :doc:`/build`. One can also run all unit tests using ctest tool
which provides higher flexibility. For example:

.. code-block:: bash

  ctest --verbose

If you need to debug errors on Windows using the debugger from VS, you can append the gtest flags in `test_main.cc`:

.. code-block::

  ::testing::GTEST_FLAG(filter) = "Suite.Test";
  ::testing::GTEST_FLAG(repeat) = 10;


***********************************************
Sanitizers: Detect memory errors and data races
***********************************************

By default, sanitizers are bundled in GCC and Clang/LLVM. One can enable sanitizers with
GCC >= 4.8 or LLVM >= 3.1, But some distributions might package sanitizers separately.
Here is a list of supported sanitizers with corresponding library names:

- Address sanitizer: libasan
- Undefined sanitizer: libubsan
- Leak sanitizer:    liblsan
- Thread sanitizer:  libtsan

Memory sanitizer is exclusive to LLVM, hence not supported in XGBoost.  With latest
compilers like gcc-9, when sanitizer flags are specified, the compiler driver should be
able to link the runtime libraries automatically.

How to build XGBoost with sanitizers
====================================
One can build XGBoost with sanitizer support by specifying -DUSE_SANITIZER=ON.
By default, address sanitizer and leak sanitizer are used when you turn the
USE_SANITIZER flag on.  You can always change the default by providing a
semicolon separated list of sanitizers to ENABLED_SANITIZERS.  Note that thread
sanitizer is not compatible with the other two sanitizers.

.. code-block:: bash

  cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;undefined" /path/to/xgboost

By default, CMake will search regular system paths for sanitizers, you can also
supply a specified SANITIZER_PATH.

.. code-block:: bash

  cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;undefined" \
  -DSANITIZER_PATH=/path/to/sanitizers /path/to/xgboost

How to use sanitizers with CUDA support
=======================================
Running XGBoost on CUDA with address sanitizer (asan) will raise memory error.
To use asan with CUDA correctly, you need to configure asan via ASAN_OPTIONS
environment variable:

.. code-block:: bash

  ASAN_OPTIONS=protect_shadow_gap=0 ${BUILD_DIR}/testxgboost


Other sanitizer runtime options
===============================

By default undefined sanitizer doesn't print out the backtrace. You can enable it by
exporting environment variable:

.. code-block::

  UBSAN_OPTIONS=print_stacktrace=1 ${BUILD_DIR}/testxgboost

For details, please consult `official documentation <https://github.com/google/sanitizers/wiki>`_ for sanitizers.
