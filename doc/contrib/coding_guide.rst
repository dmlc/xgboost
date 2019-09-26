################
Coding Guideline
################

**Contents**

.. contents::
  :backlinks: none
  :local:

********************
C++ Coding Guideline
********************
- Follow `Google style for C++ <https://google.github.io/styleguide/cppguide.html>`_, with two exceptions:

  * Each line of text may contain up to 100 characters.
  * The use of C++ exceptions is allowed.

- Use C++11 features such as smart pointers, braced initializers, lambda functions, and ``std::thread``.
- Use Doxygen to document all the interface code.
- We have a series of automatic checks to ensure that all of our codebase complies with the Google style. Before submitting your pull request, you are encouraged to run the style checks on your machine. See :ref:`running_checks_locally`.

***********************
Python Coding Guideline
***********************
- Follow `PEP 8: Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. We use PyLint to automatically enforce PEP 8 style across our Python codebase. Before submitting your pull request, you are encouraged to run PyLint on your machine. See :ref:`running_checks_locally`.
- Docstrings should be in `NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

.. _running_checks_locally:

******************
R Coding Guideline
******************

Code Style
==========
- We follow Google's C++ Style guide for C++ code.

  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.

- You can check the style of the code by typing the following command at root folder.

  .. code-block:: bash

    make rcpplint

- When needed, you can disable the linter warning of certain line with ``// NOLINT(*)`` comments.
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
See :ref:`release`.

Registering native routines in R
================================
According to `R extension manual <https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines>`_,
it is good practice to register native routines and to disable symbol search. When any changes or additions are made to the
C++ interface of the R package, please make corresponding changes in ``src/init.c`` as well.

*********************************
Running Formatting Checks Locally
*********************************

Once you submit a pull request to `dmlc/xgboost <https://github.com/dmlc/xgboost>`_, we perform
two automatic checks to enforce coding style conventions. To expedite the code review process, you are encouraged to run the checks locally on your machine prior to submitting your pull request.

Linter
======
We use `pylint <https://github.com/PyCQA/pylint>`_ and `cpplint <https://github.com/cpplint/cpplint>`_ to enforce style convention and find potential errors. Linting is especially useful for Python, as we can catch many errors that would have otherwise occured at run-time.

To run this check locally, run the following command from the top level source tree:

.. code-block:: bash

  cd /path/to/xgboost/
  make lint

This command requires the Python packages pylint and cpplint.

Clang-tidy
==========
`Clang-tidy <https://clang.llvm.org/extra/clang-tidy/>`_ is an advance linter for C++ code, made by the LLVM team. We use it to conform our C++ codebase to modern C++ practices and conventions.

To run this check locally, run the following command from the top level source tree:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py

Also, the script accepts two optional integer arguments, namely ``--cpp`` and ``--cuda``. By default they are both set to 1, meaning that both C++ and CUDA code will be checked. If the CUDA toolkit is not installed on your machine, you'll encounter an error. To exclude CUDA source from linting, use:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py --cuda=0

Similarly, if you want to exclude C++ source from linting:

.. code-block:: bash

  cd /path/to/xgboost/
  python3 tests/ci_build/tidy.py --cpp=0

