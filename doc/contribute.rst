#####################
Contribute to XGBoost
#####################
XGBoost has been developed and used by a group of active community members.
Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.

- Please add your name to `CONTRIBUTORS.md <https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md>`_ after your patch has been merged.
- Please also update `NEWS.md <https://github.com/dmlc/xgboost/blob/master/NEWS.md>`_ to add note on your changes to the API or XGBoost documentation.

**Guidelines**

* `Submit Pull Request`_
* `Running Formatting Checks Locally`_

  - `Linter`_
  - `Clang-tidy`_
  - `Running checks inside a Docker container (Recommended)`_

* `Running Unit Tests Locally`_

  - :ref:`python_tests_pytest`
  - `Google Test`_
  - `Running tests inside a Docker container (Recommended)`_

* `Git Workflow Howtos`_

  - `How to resolve conflict with master`_
  - `How to combine multiple commits into one`_
  - `What is the consequence of force push`_

* `Documents`_
* `Sanitizers`_
* `Examples`_
* `Core Library`_
* `Python Package`_
* `R Package`_

*******************
Submit Pull Request
*******************

* Before submit, please rebase your code on the most recent version of master, you can do it by

  .. code-block:: bash

    git remote add upstream https://github.com/dmlc/xgboost
    git fetch upstream
    git rebase upstream/master

* If you have multiple small commits,
  it might be good to merge them together(use git rebase then squash) into more meaningful groups.
* Send the pull request!

  - Fix the problems reported by automatic checks
  - If you are contributing a new module, consider add a testcase in `tests <https://github.com/dmlc/xgboost/tree/master/tests>`_.

*********************************
Running Formatting Checks Locally
*********************************

Once you submit a pull request to `dmlc/xgboost <https://github.com/dmlc/xgboost>`_, we perform
two automatic checks to enforce coding style conventions.

Linter
======
We use `pylint <https://github.com/PyCQA/pylint>`_ and `cpplint <https://github.com/cpplint/cpplint>`_ to enforce style convention and find potential errors. Linting is especially useful for Python, as we can catch many errors that would have otherwise occured at run-time.

To run this check locally, run the following command from the top level source tree:

.. code-block:: bash

  cd /path/to/xgboost/
  make lint

This command requires the Python packages pylint and cpplint.

.. note:: Having issue? Try Docker container

  If you are running into issues running the command above, consider using our Docker container. See :ref:`linting_inside_docker`.

Clang-tidy
==========
`Clang-tidy <https://clang.llvm.org/extra/clang-tidy/>`_ is an advance linter for C++ code, made by the LLVM team. We use it to conform our C++ codebase to modern C++ practices and conventions.

To run this check locally, run the following command from the top level source tree:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py --gtest-path=/path/to/google-test

where ``--gtest-path`` option specifies the full path of Google Test library.

Also, the script accepts two optional integer arguments, namely ``--cpp`` and ``--cuda``. By default they are both set to 1, meaning that both C++ and CUDA code will be checked. If the CUDA toolkit is not installed on your machine, you'll encounter an error. To exclude CUDA source from linting, use:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py --cuda=0 --gtest-path=/path/to/google-test

Similarly, if you want to exclude C++ source from linting:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py --cpp=0 --gtest-path=/path/to/google-test

.. note:: Having issue? Try Docker container

  If you are running into issues running the command above, consider using our Docker container. See :ref:`linting_inside_docker`.

.. _linting_inside_docker:

Running checks inside a Docker container (Recommended)
======================================================
If you have access to Docker on your machine, you can use a Docker container to automatically setup the right environment, so that you can be sure the right packages and dependencies will be available.

.. code-block:: bash

  tests/ci_build/ci_build.sh clang_tidy docker -it --build-arg CUDA_VERSION=9.2 \
    tests/ci_build/clang_tidy.sh
  tests/ci_build/ci_build.sh cpu docker -it make lint

This will run the formatting checks inside the same Docker container that `our testing server <https://xgboost-ci.net>`_ uses. Note that you don't need an NVIDIA GPU for this step.

**************************
Running Unit Tests Locally
**************************

.. _python_tests_pytest:

pytest
======
To run Python unit tests, first install `pytest <https://docs.pytest.org/en/latest/contents.html>`_ package:

.. code:: bash

  pip3 install --user pytest

Then compile XGBoost:

.. code:: bash

  mkdir build
  cd build
  cmake ..
  make
  cd ..

Now invoke pytest at the project root directory:

.. code:: bash

  export PYTHONPATH=./python-package
  pytest -v -s --fulltrace tests/python

In addition, to build and test CUDA code, run:

.. code:: bash

  cd build
  cmake -DUSE_CUDA=ON -DUSE_NCCL=ON ..
  make
  cd ..

  pytest -v -s --fulltrace tests/python-gpu

.. note:: Having issue? Try Docker container

  If you are running into issues running the command above, consider using our Docker container. See :ref:`running_tests_inside_docker`.

Google Test
===========
To build and run C++ unit tests, install `Google Test <https://github.com/google/googletest>`_ library with headers
and then enable tests while running CMake:

.. code-block:: bash

  mkdir build
  cd build
  cmake -DGOOGLE_TEST=ON -DGTEST_ROOT=/path/to/google-test ..
  make
  make test

To enable tests for CUDA code, add ``-DUSE_CUDA=ON`` and ``-DUSE_NCCL=ON`` (CUDA toolkit required):

.. code-block:: bash

  mkdir build
  cd build
  cmake -DGOOGLE_TEST=ON -DGTEST_ROOT=/path/to/google-test -DUSE_CUDA=ON -DUSE_NCCL=ON ..
  make
  make test

One can also run all unit test using ctest tool which provides higher flexibility. For example:

.. code-block:: bash

  ctest --verbose

.. note:: Having issue? Try Docker container

  If you are running into issues running the command above, consider using our Docker container. See :ref:`running_tests_inside_docker`.

.. _running_tests_inside_docker:

Running tests inside a Docker container (Recommended)
=====================================================
If you have access to Docker on your machine, you can use Docker containers to automatically setup the right environment, so that you can be sure the right packages and dependencies will be available.

Note that you need `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ to run CUDA code inside a Docker container.

The following commands will run the unit tests inside the same Docker containers that `our testing server <https://xgboost-ci.net>`_ uses:

.. code-block:: bash

  # Python tests without CUDA
  tests/ci_build/ci_build.sh cpu docker -it tests/ci_build/build_via_cmake.sh
  tests/ci_build/ci_build.sh cpu docker -it tests/ci_build/test_python.sh cpu

  # C++ tests without CUDA
  tests/ci_build/ci_build.sh cpu docker -it tests/ci_build/build_via_cmake.sh
  tests/ci_build/ci_build.sh cpu docker -it build/testxgboost

  # Python tests with CUDA (NVIDIA GPU required)
  tests/ci_build/ci_build.sh gpu_build docker -it --build-arg CUDA_VERSION=9.0 \
    tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_NCCL=ON
  tests/ci_build/ci_build.sh gpu nvidia-docker -it --build-arg CUDA_VERSION=9.0 \
    tests/ci_build/test_python.sh mgpu
  tests/ci_build/ci_build.sh gpu nvidia-docker -it --build-arg CUDA_VERSION=9.0 \
    tests/ci_build/test_python.sh gpu

  # C++ tests with CUDA (NVIDIA GPU required)
  tests/ci_build/ci_build.sh gpu_build docker -it --build-arg CUDA_VERSION=9.0 \
    tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_NCCL=ON
  tests/ci_build/ci_build.sh gpu nvidia-docker -it --build-arg CUDA_VERSION=9.0 \
    build/testxgboost

*******************
Git Workflow Howtos
*******************

How to resolve conflict with master
===================================
- First rebase to most recent master

  .. code-block:: bash

    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/dmlc/xgboost
    git fetch upstream
    git rebase upstream/master

- The git may show some conflicts it cannot merge, say ``conflicted.py``.

  - Manually modify the file to resolve the conflict.
  - After you resolved the conflict, mark it as resolved by

    .. code-block:: bash

      git add conflicted.py

- Then you can continue rebase by

  .. code-block:: bash

    git rebase --continue

- Finally push to your fork, you may need to force push here.

  .. code-block:: bash

    git push --force

How to combine multiple commits into one
========================================
Sometimes we want to combine multiple commits, especially when later commits are only fixes to previous ones,
to create a PR with set of meaningful commits. You can do it by following steps.

- Before doing so, configure the default editor of git if you haven't done so before.

  .. code-block:: bash

    git config core.editor the-editor-you-like

- Assume we want to merge last 3 commits, type the following commands

  .. code-block:: bash

    git rebase -i HEAD~3

- It will pop up an text editor. Set the first commit as ``pick``, and change later ones to ``squash``.
- After you saved the file, it will pop up another text editor to ask you modify the combined commit message.
- Push the changes to your fork, you need to force push.

  .. code-block:: bash

    git push --force

What is the consequence of force push
=====================================
The previous two tips requires force push, this is because we altered the path of the commits.
It is fine to force push to your own fork, as long as the commits changed are only yours.

*********
Documents
*********
* Documentation is built using sphinx.
* Each document is written in `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
* You can build document locally to see the effect.

**********
Sanitizers
**********

By default, sanitizers are bundled in GCC and Clang/LLVM. One can enable
sanitizers with GCC >= 4.8 or LLVM >= 3.1, But some distributions might package
sanitizers separately.  Here is a list of supported sanitizers with
corresponding library names:

- Address sanitizer: libasan
- Leak sanitizer:    liblsan
- Thread sanitizer:  libtsan

Memory sanitizer is exclusive to LLVM, hence not supported in XGBoost.

How to build XGBoost with sanitizers
====================================
One can build XGBoost with sanitizer support by specifying -DUSE_SANITIZER=ON.
By default, address sanitizer and leak sanitizer are used when you turn the
USE_SANITIZER flag on.  You can always change the default by providing a
semicolon separated list of sanitizers to ENABLED_SANITIZERS.  Note that thread
sanitizer is not compatible with the other two sanitizers.

  .. code-block:: bash

    cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak" /path/to/xgboost

By default, CMake will search regular system paths for sanitizers, you can also
supply a specified SANITIZER_PATH.

  .. code-block:: bash

    cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak" \
    -DSANITIZER_PATH=/path/to/sanitizers /path/to/xgboost

How to use sanitizers with CUDA support
=======================================
Runing XGBoost on CUDA with address sanitizer (asan) will raise memory error.
To use asan with CUDA correctly, you need to configure asan via ASAN_OPTIONS
environment variable:

  .. code-block:: bash

    ASAN_OPTIONS=protect_shadow_gap=0 ${BUILD_DIR}/testxgboost

For details, please consult `official documentation <https://github.com/google/sanitizers/wiki>`_ for sanitizers.

********
Examples
********
* Usecases and examples will be in `demo <https://github.com/dmlc/xgboost/tree/master/demo>`_.
* We are super excited to hear about your story, if you have blogposts,
  tutorials code solutions using XGBoost, please tell us and we will add
  a link in the example pages.

************
Core Library
************
- Follow `Google style for C++ <https://google.github.io/styleguide/cppguide.html>`_.
- Use C++11 features such as smart pointers, braced initializers, lambda functions, and ``std::thread``.
- We use Doxygen to document all the interface code.
- You can reproduce the linter checks by running ``make lint``

**************
Python Package
**************
- Always add docstring to the new functions in numpydoc format.
- You can reproduce the linter checks by typing ``make lint``

*********
R Package
*********

Code Style
==========
- We follow Google's C++ Style guide for C++ code.

  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.

- You can check the style of the code by typing the following command at root folder.

  .. code-block:: bash

    make rcpplint

- When needed, you can disable the linter warning of certain line with ```// NOLINT(*)``` comments.
- We use `roxygen <https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html>`_ for documenting the R package.

Rmarkdown Vignettes
===================
Rmarkdown vignettes are placed in `R-package/vignettes <https://github.com/dmlc/xgboost/tree/master/R-package/vignettes>`_.
These Rmarkdown files are not compiled. We host the compiled version on `doc/R-package <https://github.com/dmlc/xgboost/tree/master/doc/R-package>`_.

The following steps are followed to add a new Rmarkdown vignettes:

- Add the original rmarkdown to ``R-package/vignettes``.
- Modify ``doc/R-package/Makefile`` to add the markdown files to be build.
- Clone the `dmlc/web-data <https://github.com/dmlc/web-data>`_ repo to folder ``doc``.
- Now type the following command on ``doc/R-package``:

  .. code-block:: bash

    make the-markdown-to-make.md

- This will generate the markdown, as well as the figures in ``doc/web-data/xgboost/knitr``.
- Modify the ``doc/R-package/index.md`` to point to the generated markdown.
- Add the generated figure to the ``dmlc/web-data`` repo.

  - If you already cloned the repo to doc, this means ``git add``

- Create PR for both the markdown and ``dmlc/web-data``.
- You can also build the document locally by typing the following command at the ``doc`` directory:

  .. code-block:: bash

    make html

The reason we do this is to avoid exploded repo size due to generated images.

R package versioning
====================
Since version 0.6.4.3, we have adopted a versioning system that uses x.y.z (or ``core_major.core_minor.cran_release``)
format for CRAN releases and an x.y.z.p (or ``core_major.core_minor.cran_release.patch``) format for development patch versions.
This approach is similar to the one described in Yihui Xie's
`blog post on R Package Versioning <https://yihui.name/en/2013/06/r-package-versioning/>`_,
except we need an additional field to accomodate the x.y core library version.

Each new CRAN release bumps up the 3rd field, while developments in-between CRAN releases
would be marked by an additional 4th field on the top of an existing CRAN release version.
Some additional consideration is needed when the core library version changes.
E.g., after the core changes from 0.6 to 0.7, the R package development version would become 0.7.0.1, working towards
a 0.7.1 CRAN release. The 0.7.0 would not be released to CRAN, unless it would require almost no additional development.

Registering native routines in R
================================
According to `R extension manual <https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines>`_,
it is good practice to register native routines and to disable symbol search. When any changes or additions are made to the
C++ interface of the R package, please make corresponding changes in ``src/init.c`` as well.
