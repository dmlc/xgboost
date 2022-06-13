####################################
Automated testing in XGBoost project
####################################

This document collects tips for using the Continuous Integration (CI) service of the XGBoost
project.

**Contents**

.. contents::
  :backlinks: none
  :local:

**************
GitHub Actions
**************
The configuration files are located under the directory
`.github/workflows <https://github.com/dmlc/xgboost/tree/master/.github/workflows>`_.

Most of the tests listed in the configuration files run automatically for every incoming pull
requests and every update to branches. A few tests however require manual activation:

* R tests with ``noLD`` option: Run R tests using a custom-built R with compilation flag
  ``--disable-long-double``. See `this page <https://blog.r-hub.io/2019/05/21/nold/>`_ for more
  details about noLD. This is a requirement for keeping XGBoost on CRAN (the R package index).
  To invoke this test suite for a particular pull request, simply add a review comment
  ``/gha run r-nold-test``. (Ordinary comment won't work. It needs to be a review comment.)

GitHub Actions is also used to build Python wheels targeting MacOS Intel and Apple Silicon. See
`.github/workflows/python_wheels.yml
<https://github.com/dmlc/xgboost/tree/master/.github/workflows/python_wheels.yml>`_. The
``python_wheels`` pipeline sets up environment variables prefixed ``CIBW_*`` to indicate the target
OS and processor. The pipeline then invokes the script ``build_python_wheels.sh``, which in turns
calls ``cibuildwheel`` to build the wheel. The ``cibuildwheel`` is a library that sets up a
suitable Python environment for each OS and processor target. Since we don't have Apple Silion
machine in GitHub Actions, cross-compilation is needed; ``cibuildwheel`` takes care of the complex
task of cross-compiling a Python wheel. (Note that ``cibuildwheel`` will call
``setup.py bdist_wheel``. Since XGBoost has a native library component, ``setup.py`` contains
a glue code to call CMake and a C++ compiler to build the native library on the fly.)

*******************************
Reproducing errors from Jenkins
*******************************

It is often useful to reproduce the particular testing environment from our Jenkins server for
the purpose of troubleshooting a failing test. We use Docker containers heavily to package
the testing environment, so you can use Docker to reproduce it on your own machine.

1. Install Docker: https://docs.docker.com/engine/install/ubuntu/
2. Install NVIDIA Docker runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian
   The runtime lets you access NVIDIA GPUs inside a Docker container.
3. In a build log, all tests are invoked via the wrapper script ``tests/ci_build/ci_build.sh``.
   Identify the test you'd like to reproduce locally, and note how the wrapper script was invoked for that test.
   The invocation should look like this:

.. code-block:: bash

  CI_DOCKER_EXTRA_PARAMS_INIT='--shm-size=4g' tests/ci_build/ci_build.sh gpu nvidia-docker \
    --build-arg CUDA_VERSION_ARG=11.0 tests/ci_build/test_python.sh mgpu --use-rmm-pool

4. You can now run the same command on your own machine. The wrapper script will automatically download and
   set up the correct Docker container(s).
