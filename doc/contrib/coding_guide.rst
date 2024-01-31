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

- Use C++17 features such as smart pointers, braced initializers, lambda functions, and ``std::thread``.
- Use Doxygen to document all the interface code.
- We have some comments around symbols imported by headers, some of those are hinted by `include-what-you-use <https://include-what-you-use.org>`_. It's not required.
- We use clang-tidy and clang-format. You can check their configuration in the root directory of the XGBoost source tree.
- We have a series of automatic checks to ensure that all of our codebase complies with the Google style. Before submitting your pull request, you are encouraged to run the style checks on your machine. See :ref:`running_checks_locally`.

***********************
Python Coding Guideline
***********************
- Follow `PEP 8: Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. We use Pylint to automatically enforce PEP 8 style across our Python codebase. Before submitting your pull request, you are encouraged to run Pylint on your machine. See :ref:`running_checks_locally`.
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

Testing R package with different compilers
==========================================

You can change the default compiler of R by changing the configuration file in home
directory. For instance, if you want to test XGBoost built with clang++ instead of g++ on
Linux, put the following in your ``~/.R/Makevars`` file:

.. code-block:: sh

  CC=clang-15
  CXX17=clang++-15

Be aware that the variable name should match with the name used by ``R CMD``:

.. code-block:: sh

  R CMD config CXX17

Registering native routines in R
================================
According to `R extension manual <https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines>`_,
it is good practice to register native routines and to disable symbol search. When any changes or additions are made to the
C++ interface of the R package, please make corresponding changes in ``src/init.c`` as well.

Generating the Package and Running Tests
========================================

The source layout of XGBoost is a bit unusual to normal R packages as XGBoost is primarily written in C++ with multiple language bindings in mind. As a result, some special cares need to be taken to generate a standard R tarball. Most of the tests are being run on CI, and as a result, the best way to see how things work is by looking at the CI configuration files (GitHub action, at the time of writing). There are helper scripts in ``tests/ci_build`` and ``R-package/tests/helper_scripts`` for running various checks including linter and making the standard tarball.

*********************************
Running Formatting Checks Locally
*********************************

Once you submit a pull request to `dmlc/xgboost <https://github.com/dmlc/xgboost>`_, we perform
two automatic checks to enforce coding style conventions. To expedite the code review process, you are encouraged to run the checks locally on your machine prior to submitting your pull request.

Linter
======
We use a combination of linters to enforce style convention and find potential errors. Linting is especially useful for scripting languages like Python, as we can catch many errors that would have otherwise occurred at run-time.

For Python scripts, `pylint <https://github.com/PyCQA/pylint>`_, `black <https://github.com/psf/black>`__ and `isort <https://github.com/PyCQA/isort>`__ are used for providing guidance on coding style, and `mypy <https://github.com/python/mypy>`__ is required for type checking. For C++, `cpplint <https://github.com/cpplint/cpplint>`_ is used along with ``clang-tidy``. For R, ``lintr`` is used.

To run checks for Python locally, install the checkers mentioned previously and run:

.. code-block:: bash

  cd /path/to/xgboost/
  python ./tests/ci_build/lint_python.py --fix

To run checks for R:

.. code-block:: bash

  cd /path/to/xgboost/
  Rscript tests/ci_build/lint_r.R $(pwd)

To run checks for cpplint locally:

.. code-block:: bash

  cd /path/to/xgboost/
  python ./tests/ci_build/lint_cpp.py


See next section for clang-tidy. For CMake scripts:

.. code-block:: bash

  bash ./tests/ci_build/lint_cmake.sh

Lastly, the linter for jvm-packages is integrated into the maven build process.


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

**********************************
Guide for handling user input data
**********************************

This is an in-comprehensive guide for handling user input data.  XGBoost has wide verity
of native supported data structures, mostly come from higher level language bindings. The
inputs ranges from basic contiguous 1 dimension memory buffer to more sophisticated data
structures like columnar data with validity mask.  Raw input data can be used in 2 places,
firstly it's the construction of various ``DMatrix``, secondly it's the in-place
prediction.  For plain memory buffer, there's not much to discuss since it's just a
pointer with a size. But for general n-dimension array and columnar data, there are many
subtleties.  XGBoost has 3 different data structures for handling optionally masked arrays
(tensors), for consuming user inputs ``ArrayInterface`` should be chosen.  There are many
existing functions that accept only plain pointer due to legacy reasons (XGBoost started
as a much simpler library and didn't care about memory usage that much back then).  The
``ArrayInterface`` is a in memory representation of ``__array_interface__`` protocol
defined by numpy or the ``__cuda_array_interface__`` defined by numba.  Following is a
check list of things to have in mind when accepting related user inputs:

- [ ] Is it strided? (identified by the ``strides`` field)
- [ ] If it's a vector, is it row vector or column vector? (Identified by both ``shape``
  and ``strides``).
- [ ] Is the data type supported? Half type and 128 integer types should be converted
  before going into XGBoost.
- [ ] Does it have higher than 1 dimension? (identified by ``shape`` field)
- [ ] Are some of dimensions trivial? (shape[dim] <= 1)
- [ ] Does it have mask? (identified by ``mask`` field)
- [ ] Can the mask be broadcasted? (unsupported at the moment)
- [ ] Is it on CUDA memory? (identified by ``data`` field, and optionally ``stream``)

Most of the checks are handled by the ``ArrayInterface`` during construction, except for
the data type issue since it doesn't know how to cast such pointers with C builtin types.
But for safety reason one should still try to write related tests for the all items. The
data type issue should be taken care of in language binding for each of the specific data
input.  For single-chunk columnar format, it's just a masked array for each column so it
should be treated uniformly as normal array. For input predictor ``X``, we have adapters
for each type of input. Some are composition of the others. For instance, CSR matrix has 3
potentially strided arrays for ``indptr``, ``indices`` and ``values``. No assumption
should be made to these components (all the check boxes should be considered). Slicing row
of CSR matrix should calculate the offset of each field based on respective strides.

For meta info like labels, which is growing both in size and complexity, we accept only
masked array at the moment (no specialized adapter).  One should be careful about the
input data shape. For base margin it can be 2 dim or higher if we have multiple targets in
the future.  The getters in ``DMatrix`` returns only 1 dimension flatten vectors at the
moment, which can be improved in the future when it's needed.

******************************
Handling of indexable elements
******************************

There are many functionalities in XGBoost which refer to indexable elements in a countable set, such as boosting rounds / iterations / trees in a model (which can be referred to by number), classes in categorical features, among others.

XGBoost, being written in C++, uses base-0 indexing and considers ranges / sequences to be inclusive of the left end but not the right one - for example, a range (0, 3) would include the first three elements, numbered 0, 1, and 2.

The Python interface uses this same logic, since this is also the way that indexing in Python works, but other languages like R have different logic. In R, indexing is base-1 and ranges / sequences are inclusive of both ends - for example, to refer to the first three elements in a sequence, the interval would be written as (1, 3), and the elements numbered 1, 2, and 3.

In order to provide a more idiomatic R interface, XGBoost adjusts its user-facing R interface to follow this and similar R conventions, but internally, it needs to convert all these numbers to the format that the C interface uses. This is made more problematic by the fact that models are meant to be serializable and loadable in other interfaces, which will have different indexing logic.

The following adjustments are made in the R interface:

- Slicing method for DMatrix, which takes an array of integers, is converted to base-0 indexing by subtracting 1 from each element. Note that this is done in the C-level wrapper function for R, unlike all other conversions which are done in R before being passed to C.
- Slicing method for Booster takes a sequence defined by start, end, and step. The R interface is made to work the same way as R's ``seq`` from the user's POV, so it always adjusts the left end by subtracting one, and depending on whether the step size ends exactly or not at the right end, will also adjust the right end to be non-inclusive in C indexing.
- Parameter ``iterationrange`` in ``predict`` is also made to behave the same way as R's ``seq``. Since it doesn't have a step size, just adjusting the left end by subtracting 1 suffices here.
- ``best_iteration``, depending on the context, might be stored as both a C-level booster attribute, and as an R attribute. Since the C-level attributes are shared across interfaces and used in prediction methods, in order to improve compatibility, it leaves this C-level attribute in base-0 indexing, but the R attribute, if present, will be adjusted to base-1 indexing. Note that the ``predict`` method in R and other interfaces will look at the C-level attribute only.
- Categorical features are defined in R as a ``factor`` type which encodes with base-1 indexing. When categorical features are passed as R ``factor`` types, the conversion is done automatically to base-0 indexing, but if the user whishes to manually supply categorical features as already-encoded integers, then those integers need to already be in base-0 encoding.
- Categorical labels for DMatrices do not undergo any extra processing - the user must supply base-0 encoded labels.
- A function to retrieve class-specific coefficients when using the linear coefficients history callback takes a class index parameter, which also does not undergo any conversion (i.e. user must pass a base-0 index), in order to match with the label logic - that is, the same class index will refer to the class encoded with that number in the DMatrix ``label`` field.

New additions to the R interface that take on indexable elements should be mindful of these conventions and try to mimic R's behavior as much as possible.
